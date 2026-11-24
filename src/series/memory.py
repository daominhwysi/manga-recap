"""Series Memory Store: Portrait Vault, 3-Tier Bounded Memory (Hot/Warm/Cold) & Human-Editable Glossary."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import yaml
from pydantic import BaseModel, Field

from src.schemas.analysis import StoryBible
from src.schemas.ingestion import JobManifest, PanelInfo
from src.schemas.script import RecapScript
from src.schemas.series import (
    SeriesCharacter,
    SeriesContext,
    SeriesGlossary,
    SeriesState,
    VolumeHandoff,
    VolumeLedgerEntry,
)
from src.schemas.timeline import AudioManifest

if TYPE_CHECKING:
    from src.agents.page_analyzer import SeedCharacter
    from src.llm.client import LLMClient


_GLOSSARY_HEADER = """# Từ điển Series (có thể sửa tay bất cứ lúc nào giữa các tập — luôn được ưu tiên tuyệt đối)
# - terminology: khóa cách dịch cố định cho chiêu thức, tổ chức, địa danh, vật phẩm (VD: "Source Term": "Bản dịch chuẩn")
# - pronoun_pairs: khóa cặp đại từ xưng hô giữa 2 nhân vật (VD: "Nhân vật A -> Nhân vật B": "cậu - cô")
# - character_name_overrides: ép đổi tên cũ/sai sang tên chuẩn mới (VD: "Tên sai": "Tên chuẩn")
# - pronunciation_map: ghi đè phiên âm cho TTS nếu cần (mặc định để trống để TTS đọc tự nhiên)
"""


def _slugify(name: str) -> str:
    norm = unicodedata.normalize("NFKD", name)
    ascii_only = norm.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
    return slug[:32] or "char"


def _is_anonymous_label(name: str) -> bool:
    low = name.strip().lower()
    return (
        not low
        or low.startswith("[")
        or "chưa rõ tên" in low
        or "bí ẩn" in low
        or "người lạ" in low
        or "vô danh" in low
    )


def _tokenize_vi(text: str) -> set[str]:
    return {t for t in re.findall(r"[^\W\d_]{3,}", text.lower(), flags=re.UNICODE)}


class _VolumeDigestLLM(BaseModel):
    summary_vi: str = ""
    key_milestones: list[str] = Field(default_factory=list)
    ending_summary: str = ""
    cliffhanger_hook: str = ""
    active_plot_threads: list[str] = Field(default_factory=list)
    resolved_threads: list[str] = Field(default_factory=list)
    new_terminology: dict[str, str] = Field(default_factory=dict)
    new_pronoun_pairs: dict[str, str] = Field(default_factory=dict)
    deceased_characters: list[str] = Field(default_factory=list)


class SeriesStore:
    """Manages persistent series state, Portrait Vault, 3-tier bounded memory, and YAML glossary."""

    def __init__(self, series_dir: Path, series_id: str) -> None:
        self.series_dir = Path(series_dir)
        self.series_id = series_id
        self.state_path = self.series_dir / "series_state.json"
        self.glossary_path = self.series_dir / "glossary.yaml"
        self.portraits_dir = self.series_dir / "portraits"
        self.compilations_dir = self.series_dir / "compilations"

    def load_or_init(
        self,
        persona: str = "kich_tinh",
        tts_voice: str = "",
    ) -> tuple[SeriesState, SeriesGlossary]:
        self.series_dir.mkdir(parents=True, exist_ok=True)
        self.portraits_dir.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            state = SeriesState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        else:
            state = SeriesState(series_id=self.series_id, persona=persona, tts_voice=tts_voice)

        glossary = self.load_glossary()
        self.apply_glossary_overrides(state, glossary)
        if not self.glossary_path.exists():
            self.save_glossary(glossary)
        return state, glossary

    def load_glossary(self) -> SeriesGlossary:
        if not self.glossary_path.exists():
            return SeriesGlossary()
        try:
            raw = yaml.safe_load(self.glossary_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return SeriesGlossary.model_validate(raw)
        except Exception:
            pass
        return SeriesGlossary()

    def save_glossary(self, glossary: SeriesGlossary) -> Path:
        self.series_dir.mkdir(parents=True, exist_ok=True)
        dumped = yaml.safe_dump(
            glossary.model_dump(),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        self.glossary_path.write_text(_GLOSSARY_HEADER + dumped, encoding="utf-8")
        return self.glossary_path

    def save_state(self, state: SeriesState) -> Path:
        self.series_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return self.state_path

    @staticmethod
    def apply_glossary_overrides(state: SeriesState, glossary: SeriesGlossary) -> None:
        """Apply human-edited glossary overrides onto in-memory SeriesState without losing portraits."""
        if glossary.series_title.strip():
            state.series_title = glossary.series_title.strip()
        if not glossary.character_name_overrides:
            return
        ov_low = {k.strip().lower(): v.strip() for k, v in glossary.character_name_overrides.items() if k.strip() and v.strip()}
        if not ov_low:
            return
        for ch in state.characters:
            old_key = ch.canonical_name.strip().lower()
            if old_key in ov_low:
                new_name = ov_low[old_key]
                if new_name != ch.canonical_name:
                    if ch.canonical_name not in ch.aliases:
                        ch.aliases.append(ch.canonical_name)
                    ch.canonical_name = new_name
            else:
                for al in list(ch.aliases):
                    if al.strip().lower() in ov_low:
                        new_name = ov_low[al.strip().lower()]
                        if new_name != ch.canonical_name:
                            if ch.canonical_name not in ch.aliases:
                                ch.aliases.append(ch.canonical_name)
                            ch.canonical_name = new_name
                        break

    def build_context(
        self,
        volume_index: int,
        total_volumes: int | None = None,
        continuity_mode: str = "episodic",
        persona: str = "kich_tinh",
        tts_voice: str = "",
    ) -> SeriesContext:
        state, glossary = self.load_or_init(persona=persona, tts_voice=tts_voice)
        mode = continuity_mode if continuity_mode in ("standalone", "episodic", "seamless") else "episodic"
        return SeriesContext(
            series_id=self.series_id,
            series_dir=self.series_dir,
            volume_index=volume_index,
            total_volumes=total_volumes,
            continuity_mode=mode,  # type: ignore[arg-type]
            state=state,
            glossary=glossary,
        )

    @staticmethod
    def get_previously_revealed_names(state: SeriesState, current_vol: int) -> set[str]:
        """Return lowercase names/aliases of characters already officially named in volumes < current_vol."""
        revealed: set[str] = set()
        for ch in state.characters:
            if ch.first_revealed_vol is not None and ch.first_revealed_vol < current_vol:
                if not _is_anonymous_label(ch.canonical_name):
                    revealed.add(ch.canonical_name.strip().lower())
                for al in ch.aliases:
                    if not _is_anonymous_label(al):
                        revealed.add(al.strip().lower())
        return revealed

    @staticmethod
    def _find_matching_series_char(
        state: SeriesState,
        name: str,
        aliases: list[str] | None = None,
    ) -> SeriesCharacter | None:
        keys = {name.strip().lower()}
        for a in aliases or []:
            if a.strip():
                keys.add(a.strip().lower())
        keys.discard("")
        for ch in state.characters:
            ch_keys = {ch.canonical_name.strip().lower()} | {a.strip().lower() for a in ch.aliases if a.strip()}
            if keys & ch_keys:
                return ch
        return None

    @classmethod
    def select_active_cast_for_volume(
        cls,
        state: SeriesState,
        current_vol: int,
        local_seed_cast: list[SeedCharacter],
        max_cards: int = 16,
    ) -> list[SeedCharacter]:
        """Build a dynamically prioritized active cast for Volume k (fixing [:12] overflow & missing portraits)."""
        from src.agents.page_analyzer import SeedCharacter as _SeedChar

        selected: list[tuple[float, _SeedChar]] = []
        seen_names: set[str] = set()
        matched_series_ids: set[str] = set()
        handoff_chars = {
            c.strip().lower()
            for c in (state.last_handoff.ending_characters if state.last_handoff else [])
            if c.strip()
        }

        # 1) Local volume discoveries first (enriched with canonical series name + Vault portrait if present)
        for lc in local_seed_cast:
            if _is_anonymous_label(lc.canonical_name):
                continue
            sc = cls._find_matching_series_char(state, lc.canonical_name, lc.aliases)
            if sc is not None:
                matched_series_ids.add(sc.char_id)
                cname = sc.canonical_name
                merged_aliases = list(dict.fromkeys([*sc.aliases, *lc.aliases]))
                if lc.canonical_name.strip().lower() != cname.strip().lower() and lc.canonical_name not in merged_aliases:
                    merged_aliases.append(lc.canonical_name)
                portrait = sc.portrait_path if (sc.portrait_path and Path(sc.portrait_path).exists()) else ""
                appearance = lc.appearance_vi or sc.appearance
                key = cname.strip().lower()
                if key not in seen_names:
                    seen_names.add(key)
                    selected.append(
                        (
                            100.0 + (10.0 if key in handoff_chars else 0.0),
                            _SeedChar(
                                canonical_name=cname,
                                aliases=merged_aliases[:5],
                                evidence_quote=lc.evidence_quote,
                                appearance_vi=appearance,
                                best_panel_id=lc.best_panel_id,
                                portrait_path=portrait,
                            ),
                        )
                    )
            else:
                key = lc.canonical_name.strip().lower()
                if key not in seen_names:
                    seen_names.add(key)
                    selected.append((95.0, lc.model_copy(deep=True)))

        # 2) Returning series characters (prioritize handoff cast & recently active cast; exclude deceased/stale)
        for sc in state.characters:
            if sc.char_id in matched_series_ids or _is_anonymous_label(sc.canonical_name):
                continue
            key = sc.canonical_name.strip().lower()
            if key in seen_names:
                continue
            in_handoff = key in handoff_chars or any(a.strip().lower() in handoff_chars for a in sc.aliases)
            vol_gap = max(0, current_vol - sc.last_seen_vol)
            if sc.status == "deceased" and not in_handoff:
                continue
            if sc.status == "inactive" and vol_gap > 2 and not in_handoff:
                continue
            if vol_gap > 4 and not in_handoff:
                continue

            has_portrait = bool(sc.portrait_path and Path(sc.portrait_path).exists())
            score = (85.0 if in_handoff else 60.0) - vol_gap * 8.0 + (5.0 if has_portrait else 0.0)
            seen_names.add(key)
            selected.append(
                (
                    score,
                    _SeedChar(
                        canonical_name=sc.canonical_name,
                        aliases=sc.aliases[:5],
                        evidence_quote=f"Đã xuất hiện từ tập {sc.first_seen_vol}",
                        appearance_vi=sc.appearance,
                        best_panel_id="",  # Never reference stale panel_ids from prior volume!
                        portrait_path=sc.portrait_path if has_portrait else "",
                    ),
                )
            )

        selected.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in selected[:max_cards]]

    @classmethod
    def save_character_portraits(
        cls,
        state: SeriesState,
        seed_cast: list[SeedCharacter],
        manifest: JobManifest,
        portraits_dir: Path,
        current_vol: int,
    ) -> int:
        """Persist cropped 188x192 face portraits into the Series Portrait Vault."""
        from src.agents.page_analyzer import _crop_face_portrait

        portraits_dir.mkdir(parents=True, exist_ok=True)
        panel_map: dict[str, PanelInfo] = {
            pnl.panel_id: pnl
            for ch in manifest.chapters
            for pg in ch.pages
            for pnl in pg.panels
        }
        saved = 0
        for sc_entry in seed_cast:
            if _is_anonymous_label(sc_entry.canonical_name) or not sc_entry.best_panel_id:
                continue
            pnl = panel_map.get(sc_entry.best_panel_id)
            if pnl is None:
                continue
            existing = cls._find_matching_series_char(state, sc_entry.canonical_name, sc_entry.aliases)
            if existing is None:
                char_id = f"c{len(state.characters) + 1:03d}_{_slugify(sc_entry.canonical_name)}"
                existing = SeriesCharacter(
                    char_id=char_id,
                    canonical_name=sc_entry.canonical_name.strip(),
                    aliases=list(dict.fromkeys(sc_entry.aliases)),
                    appearance=sc_entry.appearance_vi,
                    first_seen_vol=current_vol,
                    last_seen_vol=current_vol,
                    portrait_vol=current_vol,
                )
                state.characters.append(existing)

            # Save portrait if missing on disk or if appearance changed significantly in later volume
            needs_save = not (existing.portrait_path and Path(existing.portrait_path).exists())
            if not needs_save:
                continue
            tile = _crop_face_portrait(pnl, target_w=188, target_h=192)
            if tile is None:
                continue
            out_file = portraits_dir / f"{existing.char_id}.jpg"
            cv2.imwrite(str(out_file), tile, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            existing.portrait_path = str(out_file.resolve())
            existing.portrait_vol = current_vol
            saved += 1
        return saved

    @staticmethod
    def retrieve_bounded_memory_prompt(
        ctx: SeriesContext,
        active_characters: list[str] | None = None,
        current_text_hint: str = "",
        max_cold_callbacks: int = 3,
    ) -> str:
        """Build O(1)-bounded 3-tier series continuity context (Hot + Warm + Cold Callbacks)."""
        state = ctx.state
        glossary = ctx.glossary
        vol = ctx.volume_index
        sections: list[str] = []

        # Header
        tot_lbl = f"/{ctx.total_volumes}" if ctx.total_volumes else ""
        sections.append(
            f"### NGỮ CẢNH CHUỖI TẬP (SERIES: {state.series_title or state.series_id} | TẬP {vol}{tot_lbl} | mode={ctx.continuity_mode})"
        )

        # Tier 1: Hot Memory (Previous volume exact ending)
        if state.last_handoff and state.last_handoff.volume_index < vol:
            ho = state.last_handoff
            hot_lines = [f"**[TẦNG NÓNG — Kết tập {ho.volume_index}]**:"]
            if ho.ending_summary:
                hot_lines.append(f"- Tình huống chốt tập {ho.volume_index}: {ho.ending_summary}")
            if ho.cliffhanger_hook:
                hot_lines.append(f"- Câu hỏi/nút thắt treo cuối tập {ho.volume_index}: {ho.cliffhanger_hook}")
            if ho.ending_characters:
                hot_lines.append(f"- Nhân vật đang ở hiện trường cuối tập trước: {', '.join(ho.ending_characters[:8])}")
            if ho.last_beats_tail:
                tail_joined = " | ".join(ho.last_beats_tail[-2:])
                hot_lines.append(f"- 2 câu kết gần nhất của tập {ho.volume_index} (CHỈ để nối mạch giọng, CẤM lặp nguyên văn): {tail_joined}")
            sections.append("\n".join(hot_lines))

        # Tier 2: Warm Memory (Active threads + Locked Glossary & Pronouns)
        warm_lines: list[str] = []
        if state.active_plot_threads:
            warm_lines.append("**[TẦNG ẤM — Mạch bí ẩn/mục tiêu đang mở]**:")
            for th in state.active_plot_threads[:8]:
                warm_lines.append(f"- {th}")
        if glossary.terminology:
            terms_fmt = "; ".join(f"{k} -> {v}" for k, v in list(glossary.terminology.items())[:25])
            warm_lines.append(f"**[TỪ ĐIỂN THUẬT NGỮ CỐ ĐỊNH — BẮT BUỘC DÙNG NHẤT QUÁN]**: {terms_fmt}")
        if glossary.pronoun_pairs:
            prons_fmt = "; ".join(f"{k}: {v}" for k, v in list(glossary.pronoun_pairs.items())[:15])
            warm_lines.append(f"**[QUY ƯỚC XƯNG HÔ XUYÊN TẬP]**: {prons_fmt}")
        if warm_lines:
            sections.append("\n".join(warm_lines))

        # Tier 3: Cold Memory (Targeted callback retrieval from volumes < vol - 1)
        older_entries = [e for e in state.ledger if e.volume_index < vol - 1]
        if older_entries:
            active_low = {c.strip().lower() for c in (active_characters or []) if c.strip()}
            hint_tokens = _tokenize_vi(current_text_hint)
            scored: list[tuple[float, VolumeLedgerEntry]] = []
            for entry in older_entries:
                score = 0.0
                entry_chars = {c.strip().lower() for c in entry.characters_featured}
                score += len(active_low & entry_chars) * 3.0
                if hint_tokens:
                    entry_tokens = _tokenize_vi(
                        f"{entry.summary_vi} {' '.join(entry.key_milestones)}"
                    )
                    score += min(6.0, len(hint_tokens & entry_tokens) * 0.4)
                # Slight recency preference as tie-breaker
                score += entry.volume_index * 0.01
                scored.append((score, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            picked = [e for s, e in scored[:max_cold_callbacks] if s > 0.5]
            if not picked:
                picked = older_entries[-min(2, max_cold_callbacks) :]
            picked.sort(key=lambda e: e.volume_index)

            cold_lines = ["**[TẦNG LẠNH — Ký ức các tập cũ liên quan để gọi callback]**:"]
            for e in picked:
                ms = f" | Mốc chính: {'; '.join(e.key_milestones[:3])}" if e.key_milestones else ""
                cold_lines.append(f"- Tập {e.volume_index} ({e.title_vi or e.job_id}): {e.summary_vi[:260]}{ms}")
            sections.append("\n".join(cold_lines))

        return "\n\n".join(sections)

    def update_after_volume(
        self,
        state: SeriesState,
        glossary: SeriesGlossary,
        volume_index: int,
        job_id: str,
        input_path: Path | str,
        manifest: JobManifest,
        bible: StoryBible,
        script: RecapScript,
        audio_manifest: AudioManifest | None = None,
        output_mp4: Path | str | None = None,
        seed_cast: list[SeedCharacter] | None = None,
        llm: LLMClient | None = None,
        cues_json_path: Path | None = None,
    ) -> tuple[SeriesState, SeriesGlossary]:
        """Consolidate Volume k into persistent SeriesState, Portrait Vault, and Glossary."""
        # 1) Series metadata (lock from Vol 1 if unset)
        if not state.series_title and bible.series_title_guess:
            state.series_title = bible.series_title_guess
        if not state.genre_guess and bible.genre_guess:
            state.genre_guess = bible.genre_guess
        if not state.tone_summary and bible.tone_summary:
            state.tone_summary = bible.tone_summary
        if not glossary.series_title and state.series_title:
            glossary.series_title = state.series_title
        if audio_manifest and audio_manifest.voice and not state.tts_voice:
            state.tts_voice = audio_manifest.voice

        # 2) Persist face portraits from this volume into Portrait Vault
        if seed_cast:
            self.save_character_portraits(
                state=state,
                seed_cast=seed_cast,
                manifest=manifest,
                portraits_dir=self.portraits_dir,
                current_vol=volume_index,
            )

        # 3) Merge StoryBible characters into SeriesState (updating reveal vol & last_seen_vol)
        featured_chars: list[str] = []
        for bch in bible.characters:
            if _is_anonymous_label(bch.canonical_name):
                continue
            cname = bch.canonical_name.strip()
            featured_chars.append(cname)
            existing = self._find_matching_series_char(state, cname, bch.aliases)
            was_revealed_here = bool(bch.first_named_page_id) or (
                existing is not None and existing.first_revealed_vol is not None
            )
            if existing is None:
                char_id = f"c{len(state.characters) + 1:03d}_{_slugify(cname)}"
                state.characters.append(
                    SeriesCharacter(
                        char_id=char_id,
                        canonical_name=cname,
                        aliases=list(dict.fromkeys(bch.aliases)),
                        role=bch.role or "Nhân vật",
                        appearance=bch.appearance,
                        personality_notes=bch.personality_notes,
                        relationships=list(bch.relationships),
                        status="active",
                        first_seen_vol=volume_index,
                        last_seen_vol=volume_index,
                        first_revealed_vol=volume_index if bch.first_named_page_id else None,
                        portrait_vol=volume_index,
                    )
                )
            else:
                existing.last_seen_vol = max(existing.last_seen_vol, volume_index)
                for al in bch.aliases:
                    if al.strip() and al not in existing.aliases and al.lower() != existing.canonical_name.lower():
                        existing.aliases.append(al)
                if bch.appearance and len(bch.appearance) > len(existing.appearance):
                    existing.appearance = bch.appearance
                if bch.personality_notes and not existing.personality_notes:
                    existing.personality_notes = bch.personality_notes
                for rel in bch.relationships:
                    if rel and rel not in existing.relationships:
                        existing.relationships.append(rel)
                existing.relationships = existing.relationships[:12]
                if existing.first_revealed_vol is None and was_revealed_here:
                    existing.first_revealed_vol = volume_index

        # 4) Digest volume into Cold Ledger + Hot Handoff + Updated Threads + New Terms
        beats = script.beats
        tail_beats = [b.narration_vi.strip() for b in beats[-3:] if b.narration_vi.strip()]
        ending_mood = beats[-1].mood if beats else "tense"

        digest: _VolumeDigestLLM | None = None
        if llm is not None and llm.has_text_credentials and beats:
            try:
                script_sample = "\n".join(
                    f"[{b.beat_id}|{b.section}] {b.narration_vi}" for b in beats
                )
                digest = llm.complete_json(
                    agent_name="SeriesContinuityUpdater",
                    system_prompt=(
                        "Bạn là Trợ lý Quản trị Ký ức Chuỗi Tập (Series Continuity Archivist). "
                        "Nhiệm vụ: nén tập truyện vừa hoàn thành thành bản ghi gọn gàng để nối liền mạch cho các tập tiếp theo.\n"
                        "Trả về JSON gồm:\n"
                        "- summary_vi: tóm tắt toàn tập gọn 3-5 câu sắc sảo.\n"
                        "- key_milestones: 3-6 mốc biến cố quan trọng nhất.\n"
                        "- ending_summary: 1-2 câu mô tả chính xác hoàn cảnh ở giây cuối tập.\n"
                        "- cliffhanger_hook: 1 câu nút thắt/câu hỏi treo nối sang tập tiếp theo.\n"
                        "- active_plot_threads: tối đa 10 mạch bí ẩn/mục tiêu CÒN MỞ (gộp mới + cũ, bỏ mạch đã giải quyết).\n"
                        "- resolved_threads: các mạch bí ẩn vừa được giải quyết ở tập này.\n"
                        "- new_terminology: dict tối đa 10 thuật ngữ/tổ chức/chiêu thức đặc trưng mới (từ gốc/biến thể -> bản dịch chuẩn tiếng Việt).\n"
                        "- new_pronoun_pairs: dict tối đa 8 cặp xưng hô cố định mới ('Nhân vật A -> Nhân vật B': 'cách xưng hô').\n"
                        "- deceased_characters: danh sách tên nhân vật đã chết hẳn trong tập này (nếu có)."
                    ),
                    user_prompt=(
                        f"Series: {state.series_title or state.series_id} | Tập {volume_index}\n"
                        f"Mạch bí ẩn đang mở trước tập này: {state.active_plot_threads}\n"
                        f"Từ điển hiện tại: {glossary.terminology}\n"
                        f"Nhân vật tập này: {featured_chars[:16]}\n\n"
                        f"Toàn bộ kịch bản tập {volume_index}:\n{script_sample[:14000]}"
                    ),
                    schema_cls=_VolumeDigestLLM,
                )
            except Exception:
                digest = None

        if digest is not None and digest.summary_vi.strip():
            vol_summary = digest.summary_vi.strip()
            milestones = [m.strip() for m in digest.key_milestones if m.strip()][:6]
            ending_summary = digest.ending_summary.strip() or (tail_beats[-1] if tail_beats else "")
            cliffhanger = digest.cliffhanger_hook.strip()
            if digest.active_plot_threads:
                state.active_plot_threads = [t.strip() for t in digest.active_plot_threads if t.strip()][:12]
            resolved = [r.strip() for r in digest.resolved_threads if r.strip()][:6]
            # Merge new terminology & pronouns WITHOUT overwriting user edits already in glossary.yaml
            for k, v in digest.new_terminology.items():
                ks, vs = k.strip(), v.strip()
                if ks and vs and ks not in glossary.terminology:
                    glossary.terminology[ks] = vs
            for k, v in digest.new_pronoun_pairs.items():
                ks, vs = k.strip(), v.strip()
                if ks and vs and ks not in glossary.pronoun_pairs:
                    glossary.pronoun_pairs[ks] = vs
            deceased_low = {d.strip().lower() for d in digest.deceased_characters if d.strip()}
            for ch in state.characters:
                if ch.canonical_name.strip().lower() in deceased_low:
                    ch.status = "deceased"
        else:
            # Deterministic offline fallback
            ch_sums = [cs.summary for cs in bible.chapter_summaries if cs.summary.strip()]
            vol_summary = (
                " ".join(ch_sums[:3])[:420]
                if ch_sums
                else " ".join(b.narration_vi for b in beats[:3])[:420]
            )
            milestones = [ev for cs in bible.chapter_summaries for ev in cs.key_events][:5]
            if not milestones and beats:
                milestones = [beats[0].narration_vi[:120], beats[-1].narration_vi[:120]]
            ending_summary = tail_beats[-1] if tail_beats else vol_summary[:180]
            cliffhanger = bible.ongoing_plot_threads[0] if bible.ongoing_plot_threads else ""
            merged_threads = list(dict.fromkeys([*state.active_plot_threads, *bible.ongoing_plot_threads]))
            state.active_plot_threads = merged_threads[-12:]
            resolved = []

        # 5) Music palette continuity (lock genre_prefs on Vol 1 + track last BGM IDs)
        last_bgm_id = ""
        if cues_json_path and cues_json_path.exists():
            try:
                cdata = json.loads(cues_json_path.read_text(encoding="utf-8"))
                if not state.music_genre_prefs and isinstance(cdata.get("genre_preferences"), dict):
                    state.music_genre_prefs = {
                        str(k): float(v) for k, v in cdata["genre_preferences"].items()
                    }
                cues_list = cdata.get("cues", [])
                if isinstance(cues_list, list) and cues_list:
                    used_ids = [str(c.get("track_id", "")).strip() for c in cues_list if c.get("track_id")]
                    if used_ids:
                        last_bgm_id = used_ids[-1]
                        state.recent_bgm_track_ids = list(dict.fromkeys([used_ids[0], used_ids[-1]]))[:4]
            except Exception:
                pass
        if not state.music_genre_prefs and state.tone_summary:
            from src.audio.music_director import infer_genre_preferences

            state.music_genre_prefs = infer_genre_preferences(
                tone_summary=state.tone_summary,
                title=state.series_title,
            )

        # 6) Record Cold Ledger entry (upsert by volume_index so re-running a volume never duplicates)
        dur_sec = round((audio_manifest.total_duration_ms / 1000.0), 2) if audio_manifest else 0.0
        entry = VolumeLedgerEntry(
            volume_index=volume_index,
            job_id=job_id,
            input_path=str(input_path),
            title_vi=script.title_vi,
            summary_vi=vol_summary,
            key_milestones=milestones,
            characters_featured=featured_chars[:12],
            resolved_threads=resolved,
            duration_sec=dur_sec,
            output_mp4=str(output_mp4) if output_mp4 else "",
        )
        state.ledger = [e for e in state.ledger if e.volume_index != volume_index]
        state.ledger.append(entry)
        state.ledger.sort(key=lambda e: e.volume_index)

        # 7) Update Hot Handoff if this is the latest volume (or replacing the latest handoff)
        if state.last_handoff is None or volume_index >= state.last_handoff.volume_index:
            state.last_handoff = VolumeHandoff(
                volume_index=volume_index,
                job_id=job_id,
                ending_summary=ending_summary,
                cliffhanger_hook=cliffhanger,
                last_beats_tail=tail_beats[-3:],
                ending_mood=ending_mood,
                ending_characters=featured_chars[:8],
                last_bgm_track_id=last_bgm_id,
            )

        self.apply_glossary_overrides(state, glossary)
        self.save_state(state)
        self.save_glossary(glossary)
        return state, glossary
