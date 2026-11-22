"""Stage 2b: PageIndex Tree Builder (Vectorless hierarchical index + MEMORY.md)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.llm.client import LLMClient
from src.schemas.analysis import (
    ChapterSummary,
    CharacterProfile,
    PageAnalysis,
    StoryBible,
)
from src.schemas.ingestion import JobManifest, PanelInfo


_BACKMATTER_KEYWORDS = (
    "translator's notes",
    "chú thích của dịch giả",
    "ghi chú của dịch giả",
    "afterword",
    "author's notes",
    "lời bạt",
    "hậu ký",
    "tâm sự của tác giả",
    "ghi chú của tác giả",
    "thông tin xuất bản",
    "bản quyền",
    "special thanks",
)

_FRONTMATTER_KEYWORDS = (
    "[phụ lục]",
    "trang trắng",
    "trang lót",
    "mục lục",
    "table of contents",
    "trang bìa",
    "bìa tập",
    "trang tiêu đề",
    "trang mở đầu chương",
    "contents",
)


def _is_backmatter_page(pa: PageAnalysis) -> bool:
    """Detect non-story backmatter pages (credits, translator notes, special thanks)."""
    blob = " ".join(
        [pa.page_summary, *(p.action_summary for p in pa.panels), *(p.dialogue_summary for p in pa.panels)]
    ).lower()
    if blob.startswith("[phụ lục]") or "[phụ lục]" in pa.page_summary.lower():
        return True
    return any(kw in blob for kw in _BACKMATTER_KEYWORDS)


def _is_frontmatter_page(
    pa: PageAnalysis,
    panel_meta: dict[str, PanelInfo] | None = None,
) -> bool:
    """Detect leading cover/blank/title/TOC pages before actual manga storytelling begins.

    Called only while `story` is still empty (leading prefix state), so no arbitrary
    page-index cutoff is needed: as soon as the first real story page appears, frontmatter
    filtering stops automatically.
    """
    blob = " ".join(
        [pa.page_summary, *(p.action_summary for p in pa.panels), *(p.dialogue_summary for p in pa.panels)]
    ).lower()
    if any(
        kw in blob
        for kw in ("[phụ lục]", "trang trắng", "trang lót", "mục lục", "table of contents")
    ):
        return True
    if len(pa.panels) <= 1:
        if any(kw in blob for kw in _FRONTMATTER_KEYWORDS):
            return True
        if pa.panels:
            p0 = pa.panels[0]
            has_dialogue = bool(p0.dialogue_summary.strip())
            info = panel_meta.get(p0.panel_id) if panel_meta else None
            has_text_fp = bool(info and any(fp.kind == "text" for fp in info.focal_points))
            if not has_dialogue and info is not None and not has_text_fp:
                return True
    return False


def _is_matching_spread_and_halves(spread_pg, p1, p2) -> bool:
    """Compare a landscape double-page spread against two adjacent portrait split pages."""
    if not (spread_pg.width > spread_pg.height * 1.15 and p1.height > p1.width and p2.height > p2.width):
        return False
    sp_path = Path(spread_pg.source_path)
    p1_path = Path(p1.source_path)
    p2_path = Path(p2.source_path)
    if not (sp_path.exists() and p1_path.exists() and p2_path.exists()):
        return False
    img_spread = cv2.imread(str(sp_path), cv2.IMREAD_GRAYSCALE)
    img_p1 = cv2.imread(str(p1_path), cv2.IMREAD_GRAYSCALE)
    img_p2 = cv2.imread(str(p2_path), cv2.IMREAD_GRAYSCALE)
    if img_spread is None or img_p1 is None or img_p2 is None:
        return False
    w = img_spread.shape[1]
    if w < 64:
        return False
    left_half = cv2.resize(img_spread[:, : w // 2], (96, 144), interpolation=cv2.INTER_AREA)
    right_half = cv2.resize(img_spread[:, w // 2 :], (96, 144), interpolation=cv2.INTER_AREA)
    r1 = cv2.resize(img_p1, (96, 144), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(img_p2, (96, 144), interpolation=cv2.INTER_AREA)
    rtl_diff = 0.5 * (
        float(np.mean(np.abs(right_half.astype(np.float32) - r1.astype(np.float32))))
        + float(np.mean(np.abs(left_half.astype(np.float32) - r2.astype(np.float32))))
    )
    ltr_diff = 0.5 * (
        float(np.mean(np.abs(left_half.astype(np.float32) - r1.astype(np.float32))))
        + float(np.mean(np.abs(right_half.astype(np.float32) - r2.astype(np.float32))))
    )
    return min(rtl_diff, ltr_diff) < 12.0


def _detect_duplicate_spread_pages(manifest: JobManifest) -> set[str]:
    """Identify split portrait pages duplicating an adjacent spread ([Spread,P1,P2] or [P1,P2,Spread])."""
    dup_pages: set[str] = set()
    for ch in manifest.chapters:
        pages = ch.pages
        for idx, pg in enumerate(pages):
            if pg.width <= pg.height * 1.15:
                continue
            # Case 1: [Spread, P1, P2]
            if idx + 2 < len(pages):
                p1, p2 = pages[idx + 1], pages[idx + 2]
                if p1.page_id not in dup_pages and p2.page_id not in dup_pages:
                    if _is_matching_spread_and_halves(pg, p1, p2):
                        dup_pages.add(p1.page_id)
                        dup_pages.add(p2.page_id)
                        continue
            # Case 2: [P1, P2, Spread]
            if idx >= 2:
                p1, p2 = pages[idx - 2], pages[idx - 1]
                if p1.page_id not in dup_pages and p2.page_id not in dup_pages:
                    if _is_matching_spread_and_halves(pg, p1, p2):
                        dup_pages.add(p1.page_id)
                        dup_pages.add(p2.page_id)
    return dup_pages


def _filter_story_pages(
    ch_pages: list[PageAnalysis],
    panel_meta: dict[str, PanelInfo] | None = None,
    duplicate_pages: set[str] | None = None,
) -> tuple[list[PageAnalysis], list[str]]:
    """Separate main story pages from leading frontmatter, duplicate spread splits, and trailing backmatter."""
    if len(ch_pages) <= 5:
        return ch_pages, []

    story: list[PageAnalysis] = []
    skipped: list[str] = []

    for pa in ch_pages:
        low_sum = pa.page_summary.strip().lower()
        if not story and _is_frontmatter_page(pa, panel_meta=panel_meta):
            skipped.append(pa.page_id)
            continue
        if low_sum.startswith("[phụ lục]"):
            skipped.append(pa.page_id)
            continue
        if duplicate_pages and pa.page_id in duplicate_pages:
            skipped.append(pa.page_id)
            continue
        if story and _is_backmatter_page(pa):
            skipped.append(pa.page_id)
            continue
        story.append(pa)

    if not story:
        return ch_pages, []
    return story, skipped


def _extract_page_blocks(md_text: str) -> list[tuple[str, str]]:
    """Split a scene Markdown note into [(page_id, markdown_block), ...]."""
    blocks: list[tuple[str, str]] = []
    cur_pid = ""
    cur_lines: list[str] = []
    for line in md_text.splitlines():
        m = re.match(r"^###\s+([A-Za-z0-9_]+):", line)
        if m:
            if cur_pid and cur_lines:
                blocks.append((cur_pid, "\n".join(cur_lines)))
            cur_pid = m.group(1)
            cur_lines = [line]
        elif cur_pid:
            cur_lines.append(line)
    if cur_pid and cur_lines:
        blocks.append((cur_pid, "\n".join(cur_lines)))
    return blocks


class SceneNode(BaseModel):
    node_id: str
    title: str
    chapter_id: str
    page_range: str
    summary: str
    narrative_weight: int = Field(default=3, ge=1, le=5)
    transition_type: str = "continuous"  # continuous, pov_switch, flashback, time_skip
    timeline_beats: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    notes_path: str
    page_ids: list[str] = Field(default_factory=list)
    panel_ids: list[str] = Field(default_factory=list)
    token_est: int = 0


class ArcNode(BaseModel):
    arc_id: str
    title: str
    summary: str
    scene_ids: list[str] = Field(default_factory=list)


class PageIndexTree(BaseModel):
    series_title: str
    total_pages: int
    total_scenes: int
    total_tokens_est: int
    tiers: int = 2  # 2 = Root->Scenes, 3 = Root->Arcs->Scenes
    arcs: list[ArcNode] = Field(default_factory=list)
    scenes: list[SceneNode] = Field(default_factory=list)
    skipped_pages: list[str] = Field(default_factory=list)

    def get_scene(self, node_id: str) -> SceneNode | None:
        for sc in self.scenes:
            if sc.node_id == node_id:
                return sc
        return None

    def get_scene_text(self, node_id: str) -> str:
        sc = self.get_scene(node_id)
        if not sc:
            return ""
        p = Path(sc.notes_path)
        return p.read_text(encoding="utf-8") if p.exists() else sc.summary

    def get_scene_tail_pages_md(self, node_id: str, max_pages: int = 2) -> str:
        """Return compact markdown for only the last `max_pages` pages of a scene."""
        md = self.get_scene_text(node_id)
        blocks = _extract_page_blocks(md)
        if not blocks:
            return ""
        return "\n".join(blk for _, blk in blocks[-max(1, max_pages) :])

    def get_scene_compact_summary(self, node_id: str) -> str:
        """Return compact summary + timeline_beats for callback reference without re-injecting panels."""
        sc = self.get_scene(node_id)
        if not sc:
            return ""
        beats = "\n".join(f"  * {tb}" for tb in sc.timeline_beats)
        return f"[{sc.node_id}] ({sc.page_range}) {sc.title}: {sc.summary}" + (
            f"\n{beats}" if beats else ""
        )

    def get_scene_tail_text(self, node_id: str, after_page_id: str) -> tuple[str, list[str]]:
        """Return markdown + panel_ids for pages in `node_id` strictly after `after_page_id`."""
        md = self.get_scene_text(node_id)
        blocks = _extract_page_blocks(md)
        if not blocks:
            return "", []
        page_ids = [pid for pid, _ in blocks]
        if after_page_id not in page_ids:
            return "", []
        cut_idx = page_ids.index(after_page_id) + 1
        if cut_idx >= len(blocks):
            return "", []
        tail_blocks = blocks[cut_idx:]
        tail_text = "\n".join(blk for _, blk in tail_blocks)
        panel_ids = re.findall(r"-\s+\[([A-Za-z0-9_]+)\]", tail_text)
        return tail_text, panel_ids

    def get_all_notes_text(self) -> str:
        chunks: list[str] = []
        for sc in self.scenes:
            p = Path(sc.notes_path)
            if p.exists():
                chunks.append(p.read_text(encoding="utf-8"))
        return "\n\n".join(chunks)

    def compact_outline_view(self) -> str:
        lines: list[str] = [
            f"# TREE INDEX: {self.series_title} ({self.total_pages} story pages, {self.total_scenes} scenes)"
        ]

        def _render_scene(sc: SceneNode) -> list[str]:
            chars = ", ".join(sc.characters[:8])
            out = [
                f"- [{sc.node_id}] ({sc.page_range}, weight={sc.narrative_weight}/5, {sc.transition_type}) "
                f"**{sc.title}**: {sc.summary}"
                + (f" | Nhân vật: {chars}" if chars else "")
            ]
            for tb in sc.timeline_beats:
                out.append(f"  * {tb}")
            return out

        if self.tiers == 3 and self.arcs:
            scene_map = {s.node_id: s for s in self.scenes}
            for arc in self.arcs:
                lines.append(f"\n## {arc.arc_id}: {arc.title} — {arc.summary}")
                for sid in arc.scene_ids:
                    sc = scene_map.get(sid)
                    if sc:
                        lines.extend(_render_scene(sc))
        else:
            for sc in self.scenes:
                lines.extend(_render_scene(sc))
        return "\n".join(lines)

    def scoped_outline_view(self, active_scene_ids: list[str], window: int = 1) -> str:
        """Render O(N) compact global map with detailed beats only for scenes near `active_scene_ids`."""
        idx_by_id = {sc.node_id: i for i, sc in enumerate(self.scenes)}
        active_indices = [idx_by_id[sid] for sid in active_scene_ids if sid in idx_by_id]
        detail_indices: set[int] = set()
        for a_idx in active_indices:
            for d in range(max(0, a_idx - window), min(len(self.scenes), a_idx + window + 1)):
                detail_indices.add(d)

        lines: list[str] = [
            f"# TREE INDEX (SCOPED): {self.series_title} ({self.total_pages} story pages, {self.total_scenes} scenes)"
        ]
        for idx, sc in enumerate(self.scenes):
            chars = ", ".join(sc.characters[:6])
            lines.append(
                f"- [{sc.node_id}] ({sc.page_range}, w={sc.narrative_weight}, {sc.transition_type}) "
                f"**{sc.title}**: {sc.summary}"
                + (f" | NV: {chars}" if chars and idx in detail_indices else "")
            )
            if idx in detail_indices:
                for tb in sc.timeline_beats:
                    lines.append(f"  * {tb}")
        return "\n".join(lines)


class _SceneSummaryLLM(BaseModel):
    node_id: str = ""
    title: str
    summary: str
    narrative_weight: int = Field(default=3, ge=1, le=5)
    transition_type: str = "continuous"
    timeline_beats: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)


class _BatchSceneSummaryResponse(BaseModel):
    scenes: list[_SceneSummaryLLM] = Field(default_factory=list)


class _SemanticSegmentSpec(BaseModel):
    start_page_id: str
    end_page_id: str
    narrative_weight: int = Field(default=3, ge=1, le=5)
    transition_type: str = "continuous"


class _SemanticSegmentationResponse(BaseModel):
    segments: list[_SemanticSegmentSpec] = Field(default_factory=list)


class _AliasRule(BaseModel):
    raw_name: str
    canonical_name: str
    page_ids: list[str] = Field(default_factory=list)


class _AliasResolutionResponse(BaseModel):
    rules: list[_AliasRule] = Field(default_factory=list)


def format_pages_compact_md(
    pages: list[PageAnalysis],
    panel_meta: dict[str, PanelInfo] | None = None,
) -> str:
    """Convert PageAnalysis list to ultra-compact Markdown preserving every panel_id + visual tags."""
    lines: list[str] = []
    for pa in pages:
        extra_bits: list[str] = []
        if pa.verbatim_quote_vi.strip():
            extra_bits.append(f'⚡Thoại: "{pa.verbatim_quote_vi.strip()}"')
        if pa.visual_punch.strip():
            extra_bits.append(f"👁️Hình: {pa.visual_punch.strip()}")
        extra_str = (" | " + " | ".join(extra_bits)) if extra_bits else ""
        lines.append(f"### {pa.page_id}: {pa.page_summary}{extra_str}")

        # Deduplicate repeated long '[Chưa rõ tên: ...]' labels within the same page
        page_anon_refs: dict[str, str] = {}
        for p in pa.panels:
            dlg = f' | "{p.dialogue_summary}"' if p.dialogue_summary else ""
            formatted_chars: list[str] = []
            for ch_name in p.characters_present:
                clean_ch = ch_name.strip()
                if not clean_ch:
                    continue
                if clean_ch.startswith("[") and len(clean_ch) > 16:
                    if clean_ch not in page_anon_refs:
                        ref_tag = f"?{len(page_anon_refs) + 1}"
                        page_anon_refs[clean_ch] = ref_tag
                        formatted_chars.append(f"{ref_tag}={clean_ch}")
                    else:
                        formatted_chars.append(f"[{page_anon_refs[clean_ch]}]")
                else:
                    formatted_chars.append(clean_ch)
            chars = f" [{', '.join(formatted_chars)}]" if formatted_chars else ""
            vis_tag = ""
            if panel_meta and p.panel_id in panel_meta:
                info = panel_meta[p.panel_id]
                ar = round(info.width / max(1, info.height), 2)
                has_face = any(fp.kind == "face" for fp in info.focal_points)
                has_text = any(fp.kind == "text" for fp in info.focal_points) or bool(p.dialogue_summary)
                flags = "+".join(
                    f for f, ok in (("face", has_face), ("text", has_text)) if ok
                ) or "bg"
                vis_tag = f"|ar={ar}|{flags}"
            lines.append(
                f"- [{p.panel_id}] ({p.emotion}|i{p.importance}{vis_tag}){chars} {p.action_summary}{dlg}"
            )
    return "\n".join(lines)


def _format_character_line(ch: CharacterProfile, char_origin: dict[str, str]) -> str:
    aliases = f" (bí danh: {', '.join(ch.aliases)})" if ch.aliases else ""
    origin = char_origin.get(ch.canonical_name.lower(), ch.first_seen_chapter)
    rels = f" | Quan hệ: {'; '.join(ch.relationships)}" if ch.relationships else ""
    return (
        f"- **{ch.canonical_name}**{aliases} [{ch.role}, gốc: {origin}]: "
        f"{ch.appearance} {ch.personality_notes}{rels}"
    )


def render_memory_md(bible: StoryBible, tree: PageIndexTree) -> str:
    """Render MEMORY.md with canonical character profiles, origin nodes, and plot threads."""
    char_origin: dict[str, str] = {}
    for sc in tree.scenes:
        for c in sc.characters:
            char_origin.setdefault(c.lower(), sc.node_id)

    lines = [
        f"# MEMORY: {bible.series_title_guess}",
        f"- Thể loại: {bible.genre_guess}",
        f"- Giọng điệu: {bible.tone_summary}",
        "- Lưu ý chống nhầm tên: Nếu dữ liệu từng trang ghi mô tả ngoại hình chung chung hoặc nhận nhầm nhân vật, hãy đối chiếu ngoại hình & ngữ cảnh với Bảng Nhân Vật Chuẩn bên dưới để gọi đúng tên.",
        "",
        "## Nhân vật chuẩn (Canonical Cast)",
    ]
    for ch in bible.characters:
        lines.append(_format_character_line(ch, char_origin))

    if bible.ongoing_plot_threads:
        lines.extend(["", "## Mạch bí ẩn & Plot Threads"])
        for pt in bible.ongoing_plot_threads:
            lines.append(f"- {pt}")

    return "\n".join(lines)


def render_scoped_memory_md(
    bible: StoryBible,
    tree: PageIndexTree,
    active_scene_ids: list[str],
    window: int = 1,
) -> str:
    """Render compact O(1) MEMORY keeping core cast, nearby-scene cast, and all plot threads."""
    if len(bible.characters) <= 10:
        return render_memory_md(bible, tree)

    idx_by_id = {sc.node_id: i for i, sc in enumerate(tree.scenes)}
    active_names: set[str] = set()
    for sid in active_scene_ids:
        if sid not in idx_by_id:
            continue
        c_idx = idx_by_id[sid]
        for j in range(max(0, c_idx - window), min(len(tree.scenes), c_idx + window + 1)):
            for cname in tree.scenes[j].characters:
                active_names.add(cname.strip().lower())

    char_origin: dict[str, str] = {}
    for sc in tree.scenes:
        for c in sc.characters:
            char_origin.setdefault(c.lower(), sc.node_id)

    selected: list[CharacterProfile] = []
    for idx, ch in enumerate(bible.characters):
        if idx < 6 or ch.canonical_name.strip().lower() in active_names:
            selected.append(ch)

    lines = [
        f"# MEMORY (SCOPED): {bible.series_title_guess}",
        f"- Thể loại: {bible.genre_guess} | Giọng điệu: {bible.tone_summary}",
        "## Nhân vật liên quan (Core + Local Cast)",
    ]
    for ch in selected:
        lines.append(_format_character_line(ch, char_origin))

    if bible.ongoing_plot_threads:
        lines.extend(["", "## Mạch bí ẩn & Plot Threads"])
        for pt in bible.ongoing_plot_threads:
            lines.append(f"- {pt}")

    return "\n".join(lines)


class PageIndexBuilder:
    """Builds hierarchical PageIndex tree, compact Markdown notes, and StoryBible/MEMORY.md."""

    def __init__(self, llm: LLMClient, pages_per_scene: int = 20) -> None:
        self.llm = llm
        self.pages_per_scene = max(5, pages_per_scene)

    def _resolve_character_aliases(
        self,
        page_analyses: list[PageAnalysis],
        job_dir: Path,
    ) -> list[PageAnalysis]:
        """Post-2a Alias Resolution: unify variants & '[Chưa rõ tên: ...]' tags against seed_cast."""
        if not self.llm.has_text_credentials or len(page_analyses) <= 8:
            return page_analyses

        seed_path = job_dir / "repo" / "seed_cast.json"
        seed_info = ""
        if seed_path.exists():
            try:
                seed_info = seed_path.read_text(encoding="utf-8")[:6000]
            except Exception:
                seed_info = ""

        # Collect distinct labels with page occurrences
        label_pages: dict[str, list[str]] = {}
        unknown_contexts: list[str] = []
        for pa in page_analyses:
            seen_on_page: set[str] = set()
            for pnl in pa.panels:
                for ch in pnl.characters_present:
                    lbl = ch.strip()
                    if not lbl:
                        continue
                    label_pages.setdefault(lbl, [])
                    if pa.page_id not in label_pages[lbl]:
                        label_pages[lbl].append(pa.page_id)
                    if (
                        lbl not in seen_on_page
                        and (
                            lbl.startswith("[")
                            or "chưa rõ" in lbl.lower()
                            or "bí ẩn" in lbl.lower()
                            or "người lạ" in lbl.lower()
                        )
                    ):
                        seen_on_page.add(lbl)
                        unknown_contexts.append(f"- {pa.page_id} [{lbl}]: {pa.page_summary[:180]}")

        label_summary = "\n".join(
            f"- {lbl!r} (xuất hiện ở {len(pgs)} trang: {pgs[:6]})"
            for lbl, pgs in sorted(label_pages.items(), key=lambda kv: -len(kv[1]))[:45]
        )
        unk_block = "\n".join(unknown_contexts[:30])

        try:
            res = self.llm.complete_json(
                agent_name="AliasResolverAgent",
                system_prompt=(
                    "Bạn là biên tập viên chuẩn hóa danh tính nhân vật trước khi lập mục lục.\n"
                    "Nhiệm vụ: gom các biến thể tên gọi (tên tắt / họ tên đầy đủ) về đúng `canonical_name` "
                    "CHỈ KHI có đủ bằng chứng trong seed_cast và tóm tắt trang. "
                    "Tuyệt đối KHÔNG gộp nhầm 2 nhân vật khác nhau và KHÔNG gán tên thật toàn cục cho các nhãn mô tả tạm "
                    "'[Chưa rõ tên: ...]' ở những trang nhân vật chưa lộ danh tính. "
                    "Nếu nhãn chỉ rõ ở một số trang cụ thể, điền danh sách `page_ids`; nếu áp dụng toàn tập, để `page_ids` trống []."
                ),
                user_prompt=(
                    f"Seed cast:\n{seed_info or 'Không có'}\n\n"
                    f"Các nhãn nhân vật xuất hiện trong 02a:\n{label_summary}\n\n"
                    f"Ngữ cảnh các nhãn chưa rõ tên:\n{unk_block}"
                ),
                schema_cls=_AliasResolutionResponse,
            )
        except Exception:
            return page_analyses

        global_map: dict[str, str] = {}
        scoped_map: dict[tuple[str, str], str] = {}
        for rule in res.rules:
            raw = rule.raw_name.strip()
            canon = rule.canonical_name.strip()
            if not raw or not canon:
                continue
            if rule.page_ids:
                for pid in rule.page_ids:
                    scoped_map[(raw.lower(), pid)] = canon
            elif not raw.startswith("[") and "chưa rõ" not in raw.lower() and "bí ẩn" not in raw.lower():
                global_map[raw.lower()] = canon

        if not global_map and not scoped_map:
            return page_analyses

        # Precompile word-boundary regexes only for safe unambiguous global aliases (len >= 3)
        global_patterns: list[tuple[re.Pattern[str], str]] = []
        for raw_low, canon in sorted(global_map.items(), key=lambda kv: -len(kv[0])):
            if len(raw_low) < 3 or raw_low == canon.lower():
                continue
            pat = re.compile(rf"(?<!\w){re.escape(raw_low)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
            global_patterns.append((pat, canon))

        def _rewrite_text(text: str, page_id: str) -> str:
            if not text:
                return text
            out = text
            # Apply page-scoped mappings (e.g. exact bracketed unknown-character tags) only on allowed page_id
            for (raw_low, scoped_pid), canon in scoped_map.items():
                if scoped_pid == page_id:
                    pat_scoped = re.compile(re.escape(raw_low), flags=re.IGNORECASE | re.UNICODE)
                    out = pat_scoped.sub(canon, out)
            for pat, canon in global_patterns:
                def _sub_fn(m: re.Match[str], _canon: str = canon, _full: str = out) -> str:
                    after = _full[m.end() : m.end() + len(_canon) + 2].lower()
                    if _canon.lower() in after:
                        return m.group(0)
                    return _canon
                out = pat.sub(_sub_fn, out)
            return out

        resolved: list[PageAnalysis] = []
        for pa in page_analyses:
            pa_copy = pa.model_copy(deep=True)
            pa_copy.page_summary = _rewrite_text(pa_copy.page_summary, pa_copy.page_id)
            pa_copy.verbatim_quote_vi = _rewrite_text(pa_copy.verbatim_quote_vi, pa_copy.page_id)
            for pnl in pa_copy.panels:
                new_chars: list[str] = []
                for ch in pnl.characters_present:
                    key = ch.strip().lower()
                    canon = scoped_map.get((key, pa_copy.page_id)) or global_map.get(key) or ch.strip()
                    if canon and canon not in new_chars:
                        new_chars.append(canon)
                pnl.characters_present = new_chars
                pnl.action_summary = _rewrite_text(pnl.action_summary, pa_copy.page_id)
                pnl.dialogue_summary = _rewrite_text(pnl.dialogue_summary, pa_copy.page_id)
            resolved.append(pa_copy)
        return resolved

    @staticmethod
    def _heuristic_narrative_weight(chunk: list[PageAnalysis]) -> int:
        if not chunk:
            return 3
        high_imp = sum(1 for pa in chunk for pnl in pa.panels if pnl.importance >= 4)
        total_pnls = max(1, sum(len(pa.panels) for pa in chunk))
        quotes = sum(1 for pa in chunk if pa.verbatim_quote_vi.strip())
        ratio = (high_imp + 0.5 * quotes) / total_pnls
        if ratio >= 0.45:
            return 5
        if ratio >= 0.30:
            return 4
        if ratio >= 0.16:
            return 3
        return 2

    def _segment_chapter_pages(
        self,
        ch_pages: list[PageAnalysis],
        notes_dir: Path,
        chapter_id: str,
        force: bool = False,
    ) -> list[tuple[list[PageAnalysis], int, str]]:
        """Group chapter story pages into narrative scenes (Semantic Segmentation + Disjoint cover check)."""
        if not ch_pages:
            return []

        seg_cache = notes_dir / f"semantic_segments_{chapter_id}.json"
        page_pos = {pa.page_id: i for i, pa in enumerate(ch_pages)}
        max_chunk = max(28, self.pages_per_scene + 10)

        def _fallback_fixed() -> list[tuple[list[PageAnalysis], int, str]]:
            out: list[tuple[list[PageAnalysis], int, str]] = []
            for start in range(0, len(ch_pages), self.pages_per_scene):
                sub = ch_pages[start : start + self.pages_per_scene]
                out.append((sub, self._heuristic_narrative_weight(sub), "continuous"))
            return out

        if not self.llm.has_text_credentials or len(ch_pages) <= 30:
            return _fallback_fixed()

        raw_segments: list[_SemanticSegmentSpec] = []
        if seg_cache.exists() and not force:
            try:
                raw_segments = _SemanticSegmentationResponse.model_validate_json(
                    seg_cache.read_text(encoding="utf-8")
                ).segments
            except Exception:
                raw_segments = []

        if not raw_segments:
            digest_lines = [
                f"- {pa.page_id} (p={len(pa.panels)}, max_imp={max((p.importance for p in pa.panels), default=2)}): "
                f"{pa.page_summary[:130]}"
                for pa in ch_pages
            ]
            try:
                seg_res = self.llm.complete_json(
                    agent_name="SemanticSceneSegmenter",
                    system_prompt=(
                        "Bạn chia tập manga thành các Scene (phân cảnh tự nhiên) liền mạch theo đúng đổi bối cảnh, "
                        "đổi tuyến nhân vật, hồi tưởng (flashback) hoặc kết thúc trận đấu — KHÔNG cắt ngang giữa cao trào.\n"
                        "Quy tắc bất biến:\n"
                        "1) Các segment phải phủ 100% trang từ trang đầu tới trang cuối theo đúng thứ tự, không nhảy cóc và không chồng lặp.\n"
                        "2) Chấm `narrative_weight` (1=chuyển cảnh nhẹ, 3=diễn biến chuẩn, 5=cao trào/twist lớn) và "
                        "`transition_type` ('continuous' | 'pov_switch' | 'flashback' | 'time_skip').\n"
                        f"3) Nên gom mỗi scene khoảng 10-{self.pages_per_scene + 4} trang (chấp nhận ngắn hơn cho flashback hoặc hơi dài hơn cho đại chiến, tối đa {max_chunk} trang)."
                    ),
                    user_prompt=f"Danh sách {len(ch_pages)} trang của {chapter_id}:\n" + "\n".join(digest_lines),
                    schema_cls=_SemanticSegmentationResponse,
                )
                raw_segments = seg_res.segments
                if raw_segments:
                    seg_cache.write_text(seg_res.model_dump_json(indent=2), encoding="utf-8")
            except Exception:
                return _fallback_fixed()

        # Verify strict contiguous disjoint cover; split oversized scenes cleanly if needed
        if not raw_segments:
            return _fallback_fixed()

        validated: list[tuple[list[PageAnalysis], int, str]] = []
        expected_start = 0
        for seg in raw_segments:
            s_idx = page_pos.get(seg.start_page_id)
            e_idx = page_pos.get(seg.end_page_id)
            if s_idx is None or e_idx is None or s_idx != expected_start or e_idx < s_idx:
                return _fallback_fixed()
            sub = ch_pages[s_idx : e_idx + 1]
            weight = max(1, min(5, int(seg.narrative_weight)))
            trans = seg.transition_type if seg.transition_type in ("continuous", "pov_switch", "flashback", "time_skip") else "continuous"
            if len(sub) > max_chunk:
                for off in range(0, len(sub), self.pages_per_scene):
                    piece = sub[off : off + self.pages_per_scene]
                    validated.append((piece, weight, trans if off == 0 else "continuous"))
            else:
                validated.append((sub, weight, trans))
            expected_start = e_idx + 1

        if expected_start != len(ch_pages):
            return _fallback_fixed()
        return validated

    def build(
        self,
        manifest: JobManifest,
        page_analyses: list[PageAnalysis],
        job_dir: Path,
        force: bool = False,
        prev_job_dir: Path | None = None,
    ) -> tuple[PageIndexTree, StoryBible]:
        repo_dir = job_dir / "repo"
        notes_dir = repo_dir / "notes"
        tree_path = repo_dir / "tree_index.json"
        memory_path = repo_dir / "MEMORY.md"
        bible_path = job_dir / "02b_story_bible.json"
        pron_path = repo_dir / "pronunciation.json"

        if tree_path.exists() and bible_path.exists() and memory_path.exists() and not force:
            tree = PageIndexTree.model_validate_json(tree_path.read_text(encoding="utf-8"))
            bible = StoryBible.model_validate_json(bible_path.read_text(encoding="utf-8"))
            return tree, bible

        notes_dir.mkdir(parents=True, exist_ok=True)
        panel_meta: dict[str, PanelInfo] = {
            pnl.panel_id: pnl
            for ch in manifest.chapters
            for pg in ch.pages
            for pnl in pg.panels
        }
        resolved_analyses = self._resolve_character_aliases(page_analyses, job_dir)

        skipped_all: list[str] = []
        dup_spread_pages = _detect_duplicate_spread_pages(manifest)
        total_tokens = 0
        total_story_pages = 0

        by_ch: dict[str, list[PageAnalysis]] = {}
        for pa in resolved_analyses:
            by_ch.setdefault(pa.chapter_id, []).append(pa)

        # Segment chapters into natural scenes & prepare chunk jobs
        chunk_jobs: list[dict] = []
        scene_counter = 1
        for ch in manifest.chapters:
            raw_ch_pages = by_ch.get(ch.chapter_id, [])
            ch_pages, ch_skipped = _filter_story_pages(
                raw_ch_pages,
                panel_meta=panel_meta,
                duplicate_pages=dup_spread_pages,
            )
            skipped_all.extend(ch_skipped)
            total_story_pages += len(ch_pages)

            segments = self._segment_chapter_pages(
                ch_pages, notes_dir=notes_dir, chapter_id=ch.chapter_id, force=force
            )
            for chunk, seg_weight, seg_trans in segments:
                node_id = f"scene_{scene_counter:03d}"
                scene_counter += 1
                page_range = f"{chunk[0].page_id}..{chunk[-1].page_id}"
                md_body = format_pages_compact_md(chunk, panel_meta=panel_meta)
                note_file = notes_dir / f"{node_id}.md"
                note_file.write_text(
                    f"# {node_id} ({ch.chapter_id}, {page_range})\n\n{md_body}\n",
                    encoding="utf-8",
                )
                tok_est = max(1, len(md_body) // 4)
                total_tokens += tok_est
                chunk_jobs.append(
                    {
                        "node_id": node_id,
                        "chapter": ch,
                        "chunk": chunk,
                        "page_range": page_range,
                        "md_body": md_body,
                        "note_file": note_file,
                        "tok_est": tok_est,
                        "weight": seg_weight,
                        "transition": seg_trans,
                    }
                )

        # Summarize scenes with per-scene cache + 3-scene batching (falls back per scene cleanly)
        cached_summaries: dict[str, _SceneSummaryLLM] = {}
        uncached_jobs: list[dict] = []
        for job in chunk_jobs:
            nid = job["node_id"]
            sc_cache = notes_dir / f"{nid}.summary.json"
            if sc_cache.exists() and not force:
                try:
                    cached_summaries[nid] = _SceneSummaryLLM.model_validate_json(
                        sc_cache.read_text(encoding="utf-8")
                    )
                    continue
                except Exception:
                    pass
            uncached_jobs.append(job)

        def _summarize_single(job: dict) -> _SceneSummaryLLM | None:
            nid = job["node_id"]
            if not (self.llm.has_text_credentials and len(page_analyses) > 30):
                return None
            try:
                sc_sum = self.llm.complete_json(
                    agent_name="PageIndexSceneSummarizer",
                    system_prompt=(
                        "Bạn lập mục lục phân đoạn manga đầy đủ từ đầu đến cuối phân đoạn.\n"
                        "BẮT BUỘC:\n"
                        "1) `node_id`: giữ đúng mã phân đoạn.\n"
                        "2) `summary`: tóm tắt tổng quan 1-2 câu (40-70 từ) phủ trọn từ trang đầu tới trang cuối.\n"
                        "3) `timeline_beats`: danh sách 4-5 gạch đầu dòng theo đúng thứ tự thời gian kèm mốc trang "
                        "(ví dụ: '[p101-p105] ...', '[p106-p110] ...'), TUYỆT ĐỐI KHÔNG bỏ sót nửa cuối và trang cuối.\n"
                        "4) `narrative_weight`: số nguyên 1..5 (1=lướt nhanh, 3=chuẩn, 5=cao trào/twist); "
                        "`transition_type`: 'continuous' | 'pov_switch' | 'flashback' | 'time_skip'."
                    ),
                    user_prompt=f"Đoạn {nid} ({job['page_range']}):\n{job['md_body'][:48000]}",
                    schema_cls=_SceneSummaryLLM,
                )
                sc_sum.node_id = nid
                (notes_dir / f"{nid}.summary.json").write_text(
                    sc_sum.model_dump_json(indent=2), encoding="utf-8"
                )
                return sc_sum
            except Exception:
                return None

        def _summarize_batch(batch_jobs: list[dict]) -> None:
            if len(batch_jobs) == 1:
                got = _summarize_single(batch_jobs[0])
                if got is not None:
                    cached_summaries[batch_jobs[0]["node_id"]] = got
                return
            joined = "\n\n".join(
                f"=== SCENE {j['node_id']} ({j['page_range']}, gợi ý weight={j['weight']}, {j['transition']}) ===\n{j['md_body'][:18000]}"
                for j in batch_jobs
            )
            expected_ids = [j["node_id"] for j in batch_jobs]
            try:
                res = self.llm.complete_json(
                    agent_name="PageIndexSceneSummarizer",
                    system_prompt=(
                        "Bạn lập mục lục đầy đủ cho từng phân đoạn manga trong batch.\n"
                        "BẮT BUỘC trả về đúng 1 mục trong `scenes` cho mỗi `node_id` được giao, giữ chính xác `node_id`.\n"
                        "Mỗi mục gồm: `node_id`, `title`, `summary` (40-70 từ phủ trọn đầu đến cuối), "
                        "`timeline_beats` (4-5 mốc kèm `[trang]`, KHÔNG sót trang cuối), `characters`, "
                        "`narrative_weight` (1-5), `transition_type` ('continuous'|'pov_switch'|'flashback'|'time_skip')."
                    ),
                    user_prompt=f"Hãy tóm tắt đầy đủ các scene sau ({expected_ids}):\n\n{joined}",
                    schema_cls=_BatchSceneSummaryResponse,
                )
                by_id = {s.node_id: s for s in res.scenes if s.node_id in expected_ids}
                for j in batch_jobs:
                    nid = j["node_id"]
                    if nid in by_id:
                        cached_summaries[nid] = by_id[nid]
                        (notes_dir / f"{nid}.summary.json").write_text(
                            by_id[nid].model_dump_json(indent=2), encoding="utf-8"
                        )
                    else:
                        single = _summarize_single(j)
                        if single is not None:
                            cached_summaries[nid] = single
            except Exception:
                for j in batch_jobs:
                    single = _summarize_single(j)
                    if single is not None:
                        cached_summaries[j["node_id"]] = single

        if self.llm.has_text_credentials and len(page_analyses) > 30 and uncached_jobs:
            job_batches = [uncached_jobs[i : i + 3] for i in range(0, len(uncached_jobs), 3)]
            if len(job_batches) > 1:
                with ThreadPoolExecutor(max_workers=min(3, len(job_batches))) as pool:
                    list(pool.map(_summarize_batch, job_batches))
            else:
                _summarize_batch(job_batches[0])

        scenes: list[SceneNode] = []
        for job in chunk_jobs:
            node_id = job["node_id"]
            ch = job["chapter"]
            chunk: list[PageAnalysis] = job["chunk"]
            page_range = job["page_range"]
            note_file = job["note_file"]
            tok_est = job["tok_est"]

            pg_ids = [pa.page_id for pa in chunk]
            pids = [pnl.panel_id for pa in chunk for pnl in pa.panels]

            chars_seen: list[str] = []
            for pa in chunk:
                for pnl in pa.panels:
                    for c in pnl.characters_present:
                        if c and c not in chars_seen:
                            chars_seen.append(c)

            timeline_beats: list[str] = []
            weight = job["weight"]
            trans_type = job["transition"]
            sc_sum = cached_summaries.get(node_id)
            if sc_sum is not None:
                title = sc_sum.title
                summary = sc_sum.summary
                timeline_beats = sc_sum.timeline_beats[:6]
                weight = max(1, min(5, int(sc_sum.narrative_weight or weight)))
                if sc_sum.transition_type in ("continuous", "pov_switch", "flashback", "time_skip"):
                    trans_type = sc_sum.transition_type
                if sc_sum.characters:
                    chars_seen = sc_sum.characters
            else:
                title = f"{ch.title} ({page_range})"
                summary = (
                    f"{chunk[0].page_summary} ... {chunk[-1].page_summary}"[:260]
                    if len(page_analyses) > 30
                    else " | ".join(pa.page_summary for pa in chunk[:2])[:220]
                )
                if len(chunk) >= 4:
                    timeline_beats = [
                        f"[{chunk[0].page_id}] {chunk[0].page_summary}",
                        f"[{chunk[len(chunk) // 2].page_id}] {chunk[len(chunk) // 2].page_summary}",
                        f"[{chunk[-1].page_id}] {chunk[-1].page_summary}",
                    ]

            scenes.append(
                SceneNode(
                    node_id=node_id,
                    title=title,
                    chapter_id=ch.chapter_id,
                    page_range=page_range,
                    summary=summary,
                    narrative_weight=weight,
                    transition_type=trans_type,
                    timeline_beats=timeline_beats,
                    characters=chars_seen[:12],
                    notes_path=str(note_file.resolve()),
                    page_ids=pg_ids,
                    panel_ids=pids,
                    token_est=tok_est,
                )
            )

        # Build 3-tier arcs automatically if > 40 scenes (for 1,000+ page scale)
        arcs: list[ArcNode] = []
        tiers = 2
        if len(scenes) > 40:
            tiers = 3
            arc_size = 12
            for a_idx in range(0, len(scenes), arc_size):
                grp = scenes[a_idx : a_idx + arc_size]
                arcs.append(
                    ArcNode(
                        arc_id=f"arc_{len(arcs) + 1:02d}",
                        title=f"Phần {len(arcs) + 1} ({grp[0].page_range.split('..')[0]}..{grp[-1].page_range.split('..')[-1]})",
                        summary=" / ".join(s.title for s in grp[:3]),
                        scene_ids=[s.node_id for s in grp],
                    )
                )

        tree = PageIndexTree(
            series_title=manifest.job_id,
            total_pages=total_story_pages or len(page_analyses),
            total_scenes=len(scenes),
            total_tokens_est=total_tokens,
            tiers=tiers,
            arcs=arcs,
            scenes=scenes,
            skipped_pages=skipped_all,
        )

        # Gather seed_cast.json & optional previous-volume StoryBible for rich character continuity
        seed_path = repo_dir / "seed_cast.json"
        seed_cast_txt = ""
        if seed_path.exists():
            try:
                seed_cast_txt = seed_path.read_text(encoding="utf-8")[:5000]
            except Exception:
                seed_cast_txt = ""

        prev_bible: StoryBible | None = None
        if prev_job_dir is not None and (prev_job_dir / "02b_story_bible.json").exists():
            try:
                prev_bible = StoryBible.model_validate_json(
                    (prev_job_dir / "02b_story_bible.json").read_text(encoding="utf-8")
                )
            except Exception:
                prev_bible = None

        prev_bible_note = (
            f"\nHồ sơ tập trước để giữ đồng bộ tên & phiên âm:\n{prev_bible.model_dump_json(indent=2)[:4000]}\n"
            if prev_bible
            else ""
        )

        # Build StoryBible + TTS Pronunciation Glossary
        if self.llm.has_text_credentials:
            bible = self.llm.complete_json(
                agent_name="StoryBibleAgent",
                system_prompt=(
                    "Bạn là StoryBibleAgent quản lý hồ sơ nhân vật chuẩn (canonical_name, aliases, role, "
                    "appearance, personality_notes, relationships), mạch cốt truyện và bảng phiên âm `pronunciation_map` "
                    "cho TTS tiếng Việt (ánh xạ các tên riêng/thuật ngữ tiếng Anh/Nhật khó đọc sang cách đọc tiếng Việt tự nhiên, dễ nghe, không dấu gạch nối thừa)."
                ),
                user_prompt=(
                    f"Series: {manifest.job_id}\n"
                    f"Seed Cast (ngoại hình & bằng chứng thoại từ Vision):\n{seed_cast_txt or 'Không có'}\n"
                    f"{prev_bible_note}\n"
                    f"Mục lục chi tiết từng phân đoạn:\n{tree.compact_outline_view()}\n\n"
                    "Hãy tổng hợp StoryBible hoàn chỉnh dạng JSON (series_title_guess, genre_guess, tone_summary, "
                    "characters, chapter_summaries, ongoing_plot_threads, pronunciation_map)."
                ),
                schema_cls=StoryBible,
            )
            if prev_bible and prev_bible.pronunciation_map:
                merged_pron = dict(prev_bible.pronunciation_map)
                merged_pron.update(bible.pronunciation_map)
                bible.pronunciation_map = merged_pron
        else:
            all_chars: list[CharacterProfile] = []
            seen_names: set[str] = set()
            if prev_bible:
                for ch_prev in prev_bible.characters:
                    seen_names.add(ch_prev.canonical_name.lower())
                    all_chars.append(ch_prev)
            if seed_cast_txt:
                try:
                    for sc_entry in json.loads(seed_cast_txt).get("characters", []):
                        cname = str(sc_entry.get("canonical_name", "")).strip()
                        if cname and cname.lower() not in seen_names:
                            seen_names.add(cname.lower())
                            all_chars.append(
                                CharacterProfile(
                                    canonical_name=cname,
                                    aliases=sc_entry.get("aliases", []),
                                    appearance=sc_entry.get("appearance_vi", ""),
                                )
                            )
                except Exception:
                    pass
            for sc in scenes:
                for cname in sc.characters:
                    if cname.lower() not in seen_names:
                        seen_names.add(cname.lower())
                        all_chars.append(
                            CharacterProfile(canonical_name=cname, first_seen_chapter=sc.chapter_id)
                        )
            bible = StoryBible(
                series_title_guess=manifest.job_id,
                characters=all_chars[:20],
                chapter_summaries=[
                    ChapterSummary(
                        chapter_id=ch.chapter_id,
                        title=ch.title,
                        summary=f"Tóm tắt {ch.title} ({len(by_ch.get(ch.chapter_id, []))} trang).",
                        important_panel_ids=[],
                    )
                    for ch in manifest.chapters
                ],
                pronunciation_map=dict(prev_bible.pronunciation_map) if prev_bible else {},
            )

        pron_path.write_text(
            json.dumps(bible.pronunciation_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tree_path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        memory_path.write_text(render_memory_md(bible, tree), encoding="utf-8")
        bible_path.write_text(bible.model_dump_json(indent=2), encoding="utf-8")
        return tree, bible
