"""Stage 3b Agent: Closed-Loop Critic, Read-Aloud Showrunner Pass & Pronunciation Prep."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.agents.page_index import extract_distinctive_char_patterns
from src.audio.base import apply_pronunciation_glossary
from src.llm.client import (
    LLMClient,
    extract_xml_attr,
    extract_xml_tag,
    iter_xml_blocks,
)
from src.schemas.analysis import PageAnalysis, StoryBible
from src.schemas.ingestion import JobManifest
from src.schemas.script import CriticIssue, CriticReport, RecapScript, ScriptBeat

if TYPE_CHECKING:
    from src.config import ScriptConfig


_BANNED_CLICHES = (
    "định mệnh nghiệt ngã",
    "bánh răng định mệnh",
    "không gian vỡ vụn",
    "xé toạc không gian",
    "rợn tóc gáy",
    "lưỡi dao cào xé",
    "bàn cờ chính trị đẫm máu",
    "chuyển sang đầu phần mới",
    "chuyển sang phần mới",
    "nối tiếp cao trào",
    "bước sang diễn biến tiếp theo",
    "sang hồi kết mở",
)


def _page_pos_from_panel(panel_id: str, page_order: dict[str, int]) -> int | None:
    pg = re.sub(r"_P\d+$", "", panel_id)
    return page_order.get(pg)


def _content_ngrams(text: str) -> set[str]:
    """Extract normalized word 2-grams and 3-grams to catch cross-beat content repetition."""
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    ngrams: set[str] = set()
    for n in (2, 3):
        for i in range(max(0, len(words) - n + 1)):
            ngrams.add(" ".join(words[i : i + n]))
    return ngrams


class _SeamRewrite(BaseModel):
    beat_id: str
    narration_vi: str


class _SeamPolishResponse(BaseModel):
    rewrites: list[_SeamRewrite] = Field(default_factory=list)


_SEAM_POLISH_XML_TEMPLATE = """<rewrites>
  <rewrite beat_id="b001">Câu kể văn nói tiếng Việt đã được chuốt lại mượt mà...</rewrite>
</rewrites>"""


def _parse_seam_polish_xml(xml_text: str) -> _SeamPolishResponse:
    rewrites: list[_SeamRewrite] = []
    for attrs, body in iter_xml_blocks(xml_text, "rewrite"):
        bid = extract_xml_attr(attrs, "beat_id") or extract_xml_attr(attrs, "id") or extract_xml_tag(body, "beat_id")
        txt = extract_xml_tag(body, "narration") or re.sub(r"<[^>]+>", "", body).strip()
        if bid and txt:
            rewrites.append(_SeamRewrite(beat_id=bid, narration_vi=txt))
    return _SeamPolishResponse(rewrites=rewrites)


class CriticAgent:
    """Validates hard invariants, macro rhythm distribution, and runs Closed-Loop Showrunner Auto-Repair."""

    def __init__(self, llm: LLMClient, polish_seams: bool = True) -> None:
        self.llm = llm
        self.polish_seams = polish_seams

    @staticmethod
    def _scoped_fallback_panel(
        beat_idx: int,
        total_beats: int,
        valid_ids: list[str],
        cleaned_beats: list[ScriptBeat],
    ) -> str:
        if cleaned_beats and cleaned_beats[-1].panel_ids:
            last_pid = cleaned_beats[-1].panel_ids[-1]
            if last_pid in valid_ids:
                pos = valid_ids.index(last_pid)
                return valid_ids[min(len(valid_ids) - 1, pos + 1)]
        ratio = beat_idx / max(1, total_beats)
        return valid_ids[min(len(valid_ids) - 1, int(ratio * len(valid_ids)))]

    @staticmethod
    def _evaluate_issues(
        beats: list[ScriptBeat],
        all_pages: list[str],
        page_order: dict[str, int],
        beat_to_allowed: dict[int, set[str]],
        target_total_words: int | None = None,
        bible: StoryBible | None = None,
    ) -> tuple[list[CriticIssue], set[int], list[str]]:
        """Evaluate hard invariants (error) & macro rhythm/span warnings without mutating panel_ids."""
        issues: list[CriticIssue] = []
        flagged_indices: set[int] = set()
        flagged_notes: list[str] = []
        total_beats = len(beats)
        avg_pages_per_beat = (len(all_pages) / max(1, total_beats)) if total_beats >= 6 else 2.5
        max_single_beat_span = max(3, math.ceil(avg_pages_per_beat * 1.5))
        max_coverage_gap = max(8, math.ceil(avg_pages_per_beat * 3.5))

        reveal_checks: list[tuple[str, str, int, list[re.Pattern[str]]]] = []
        if bible and bible.characters:
            canon_set = {c.canonical_name.strip().lower() for c in bible.characters if c.canonical_name.strip()}
            for ch in bible.characters:
                if ch.first_named_page_id and ch.first_named_page_id in page_order:
                    pats = extract_distinctive_char_patterns(ch, canon_set)
                    if pats:
                        reveal_checks.append(
                            (ch.canonical_name, ch.first_named_page_id, page_order[ch.first_named_page_id], pats)
                        )

        prev_page_pos: int | None = None
        beat_ngrams: list[tuple[str, set[str]]] = []
        word_counts: list[int] = []

        for idx, beat in enumerate(beats):
            kept = beat.panel_ids
            if idx in beat_to_allowed and kept:
                allowed = beat_to_allowed[idx]
                out_ids = [pid for pid in kept if pid not in allowed]
                if out_ids:
                    msg = f"{beat.beat_id} references panels outside assigned part range: {out_ids}."
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="panel_sync", message=msg)
                    )
                    flagged_indices.add(idx)
                    flagged_notes.append(msg)

            positions = [p for pid in kept if (p := _page_pos_from_panel(pid, page_order)) is not None]
            if positions and beat.beat_type != "hook_montage":
                cur_min = min(positions)
                cur_max = max(positions)
                if cur_max - cur_min > max_single_beat_span:
                    span_msg = (
                        f"{beat.beat_id} spans {cur_max - cur_min} pages "
                        f"({all_pages[cur_min]}..{all_pages[cur_max]}), risking visual cramming."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="warning", category="panel_sync", message=span_msg)
                    )
                    flagged_notes.append(span_msg)
                if prev_page_pos is not None and cur_min < prev_page_pos:
                    back_msg = (
                        f"{beat.beat_id} jumped backwards from {all_pages[prev_page_pos]} to {all_pages[cur_min]}."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="panel_sync", message=back_msg)
                    )
                    flagged_indices.add(idx)
                    flagged_notes.append(back_msg)
                if prev_page_pos is not None and (cur_min - prev_page_pos) > max_coverage_gap:
                    gap_msg = (
                        f"Page coverage gap before {beat.beat_id}: jumped from {all_pages[prev_page_pos]} "
                        f"to {all_pages[cur_min]} ({cur_min - prev_page_pos} pages skipped)."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="coverage_gap", message=gap_msg)
                    )
                    flagged_notes.append(gap_msg)
                prev_page_pos = max(prev_page_pos or 0, cur_max)

                for cname, rev_pg, rev_pos, pats in reveal_checks:
                    if rev_pos > cur_max and any(p.search(beat.narration_vi) for p in pats):
                        sp_msg = (
                            f"{beat.beat_id} gọi tên '{cname}' trước khi nhân vật lộ tên trong thoại (tại {rev_pg}) "
                            f"— hãy đổi sang mô tả ngoại hình/vai trò."
                        )
                        issues.append(
                            CriticIssue(beat_id=beat.beat_id, severity="error", category="spoiler_name", message=sp_msg)
                        )
                        flagged_indices.add(idx)
                        flagged_notes.append(sp_msg)

            cur_ng = _content_ngrams(beat.narration_vi)
            if len(cur_ng) >= 8:
                for prev_bid, prev_ng in beat_ngrams:
                    overlap = len(cur_ng & prev_ng)
                    denom = max(1, min(len(cur_ng), len(prev_ng)))
                    if overlap >= 4 and (overlap / denom) >= 0.35:
                        dup_msg = f"{beat.beat_id} possible repeated content overlapping with {prev_bid}."
                        issues.append(
                            CriticIssue(
                                beat_id=beat.beat_id,
                                severity="error",
                                category="duplication",
                                message=dup_msg,
                            )
                        )
                        flagged_indices.add(idx)
                        flagged_notes.append(dup_msg)
                        break
                beat_ngrams.append((beat.beat_id, cur_ng))

            words = len(beat.narration_vi.split())
            word_counts.append(words)
            if words > 80:
                p_msg = f"{beat.beat_id} exceeds 80 words ({words} từ); tighten phrasing for voiceover."
                issues.append(
                    CriticIssue(beat_id=beat.beat_id, severity="warning", category="pacing", message=p_msg)
                )
                flagged_indices.add(idx)
                flagged_notes.append(p_msg)

            low_text = beat.narration_vi.lower()
            for phrase in _BANNED_CLICHES:
                if phrase in low_text:
                    c_msg = f"{beat.beat_id} chứa cụm sáo rỗng '{phrase}'."
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="style", message=c_msg)
                    )
                    flagged_indices.add(idx)
                    flagged_notes.append(c_msg)

        # Macro Distribution Critique (Rule 5): check volume-level pacing variance & total word budget
        if len(word_counts) >= 12:
            mean_wc = sum(word_counts) / len(word_counts)
            std_wc = math.sqrt(sum((w - mean_wc) ** 2 for w in word_counts) / len(word_counts))
            if std_wc < 4.5:
                issues.append(
                    CriticIssue(
                        beat_id="global",
                        severity="warning",
                        category="macro_rhythm",
                        message=f"Beat rhythm is overly uniform (std={std_wc:.1f} words); vary short/long beats.",
                    )
                )

        if target_total_words and len(beats) >= 6:
            actual_words = sum(word_counts)
            if actual_words > int(target_total_words * 1.30):
                excess = actual_words - int(target_total_words * 1.22)
                b_msg = (
                    f"Total script length ({actual_words} từ) exceeds target ({target_total_words} từ) by >30% "
                    f"— hãy rút gọn bớt ~{excess} từ ở các beat dài."
                )
                issues.append(
                    CriticIssue(beat_id="global", severity="error", category="budget", message=b_msg)
                )
                longest_indices = sorted(
                    range(len(beats)),
                    key=lambda i: word_counts[i],
                    reverse=True,
                )[: min(8, len(beats))]
                for li in longest_indices:
                    flagged_indices.add(li)
                flagged_notes.append(b_msg)

        return issues, flagged_indices, flagged_notes

    def _polish_multi_part_seams(
        self,
        beats: list[ScriptBeat],
        bible: StoryBible,
        seam_indices: list[int],
        flagged_indices: set[int],
        flagged_notes: list[str],
        panel_text_lookup: dict[str, str] | None = None,
        run_global_showrunner: bool = False,
        target_total_words: int | None = None,
    ) -> list[ScriptBeat]:
        """Read-Aloud Showrunner Pass + Targeted Auto-Repair (panel_ids strictly locked in Python)."""
        if not self.polish_seams or not self.llm.has_text_credentials or len(beats) < 6:
            return beats

        seam_set = {i for i in seam_indices if 0 <= i < len(beats)}
        error_neighbors: set[int] = set()
        for fi in flagged_indices:
            for d in (fi - 1, fi, fi + 1):
                if 0 <= d < len(beats):
                    error_neighbors.add(d)

        target_indices = (
            list(range(len(beats)))
            if run_global_showrunner
            else sorted(flagged_indices | seam_set | error_neighbors)
        )
        if not target_indices:
            return beats

        seam_beat_ids = [beats[i].beat_id for i in sorted(seam_set)]
        payload_lines = ["<beats>"]
        for i in target_indices:
            b = beats[i]
            facts_attr = ""
            if i in flagged_indices and panel_text_lookup:
                facts = [panel_text_lookup[pid] for pid in b.panel_ids if pid in panel_text_lookup]
                if facts:
                    facts_txt = " | ".join(facts).replace('"', "'")
                    facts_attr = f"\n    <grounded_facts>{facts_txt}</grounded_facts>"
            payload_lines.append(
                f'  <beat id="{b.beat_id}" section="{b.section}" beat_type="{b.beat_type}" mood="{b.mood}">\n'
                f"    <narration>{b.narration_vi}</narration>{facts_attr}\n"
                "  </beat>"
            )
        payload_lines.append("</beats>")

        cast_names = [
            f"{c.canonical_name} (lộ từ {c.first_named_page_id})" if c.first_named_page_id else c.canonical_name
            for c in bible.characters
        ]
        notes_txt = "\n".join(f"- {n}" for n in flagged_notes) or "Không có lỗi nghiêm trọng — hãy chuốt văn nói mượt mà."
        budget_hint = (
            f"Tổng ngân sách mục tiêu toàn tập: ~{target_total_words} từ (cắt gọn câu thừa nếu vượt quá nhiều)."
            if target_total_words
            else ""
        )
        try:
            res = self.llm.complete_xml(
                agent_name="CriticSeamPolisher",
                system_prompt=(
                    "Bạn là Tổng Biên Tập Kịch Bản Voiceover YouTube (Read-Aloud Showrunner). CHỈ trả về các thẻ `<rewrite beat_id='...'>` cho những beat thực sự cần sửa để:\n"
                    "1) Nối mượt chuyển hồi tại `seam_beat_ids` bằng hành động/không gian tự nhiên, xóa giới thiệu lặp nhân vật. CẤM câu chuyển đoạn bình luận viên.\n"
                    "2) Khử sạch cliché AI, lặp từ nối ('ai ngờ', 'cứ tưởng', 'tới lúc này', 'thế nhưng'), bỏ tên nhân vật bị gọi sớm trước `lộ từ ...`, và sửa trùng lặp dựa ĐÚNG trên `grounded_facts`.\n"
                    "3) BẢO TOÀN TUYỆT ĐỐI giọng kể YouTuber gần gũi ('mình', 'anh em', 'các bạn'), các câu bình luận/phân tích (`beat_type='commentary'`), và nhịp beat ngắn dồn dập (6-15 từ) — tuyệt đối KHÔNG viết lại thành văn xuôi tóm tắt sách vở."
                ),
                user_prompt=(
                    f"Nhân vật chuẩn: {cast_names}\n"
                    f"Các điểm nối hồi (seam): {seam_beat_ids}\n"
                    f"{budget_hint}\n"
                    f"Lỗi phát hiện tự động:\n{notes_txt}\n\n"
                    f"Kịch bản cần rà soát ({len(target_indices)} beats):\n"
                    + "\n".join(payload_lines)
                ),
                xml_template=_SEAM_POLISH_XML_TEMPLATE,
                parse_fn=_parse_seam_polish_xml,
            )
            rewrite_map: dict[str, str] = {}
            for r in res.rewrites:
                txt = r.narration_vi.strip()
                if not txt or txt[0].islower():
                    continue
                low_txt = txt.lower()
                if any(phrase in low_txt for phrase in _BANNED_CLICHES):
                    continue
                rewrite_map[r.beat_id] = txt
            for b in beats:
                if b.beat_id in rewrite_map:
                    b.narration_vi = rewrite_map[b.beat_id]
        except Exception:
            pass
        return beats

    def review(
        self,
        script: RecapScript,
        manifest: JobManifest,
        bible: StoryBible,
        job_dir: Path,
        force: bool = False,
        script_config: ScriptConfig | None = None,
        page_analyses: list[PageAnalysis] | None = None,
    ) -> tuple[RecapScript, CriticReport]:
        final_path = job_dir / "03b_script_final.json"
        report_path = job_dir / "03b_critic_report.json"
        if final_path.exists() and report_path.exists() and not force:
            return (
                RecapScript.model_validate_json(final_path.read_text(encoding="utf-8")),
                CriticReport.model_validate_json(report_path.read_text(encoding="utf-8")),
            )

        all_pages = [pg.page_id for ch in manifest.chapters for pg in ch.pages]
        page_order = {pg_id: i for i, pg_id in enumerate(all_pages)}
        valid_ids = [
            p.panel_id
            for ch in manifest.chapters
            for pg in ch.pages
            for p in pg.panels
        ]
        valid_set = set(valid_ids)

        panel_text_lookup: dict[str, str] = {}
        if page_analyses:
            for pa in page_analyses:
                for pnl in pa.panels:
                    dlg = f' "{pnl.dialogue_summary}"' if pnl.dialogue_summary else ""
                    panel_text_lookup[pnl.panel_id] = f"{pnl.action_summary}{dlg}".strip()

        # Load real part boundaries if written by ScriptwriterAgent
        seam_indices: list[int] = []
        beat_to_allowed: dict[int, set[str]] = {}
        boundaries_path = job_dir / "repo" / "script" / "part_boundaries.json"
        if boundaries_path.exists():
            try:
                bounds = json.loads(boundaries_path.read_text(encoding="utf-8"))
                for b_idx, part_meta in enumerate(bounds):
                    end_i = int(part_meta.get("end_beat_idx", -1))
                    if b_idx < len(bounds) - 1 and end_i >= 0:
                        seam_indices.extend([end_i, end_i + 1])
                    allowed = set(part_meta.get("allowed_panels", []))
                    if allowed:
                        for idx_in in range(
                            int(part_meta.get("start_beat_idx", 0)),
                            int(part_meta.get("end_beat_idx", -1)) + 1,
                        ):
                            beat_to_allowed[idx_in] = allowed
            except Exception:
                seam_indices = []
        seam_indices = sorted(set(seam_indices))

        # Step 1: Deterministic Panel Grounding repair (guaranteed valid panel_ids)
        grounding_issues: list[CriticIssue] = []
        cleaned_beats: list[ScriptBeat] = []
        total_beats = len(script.beats)
        for idx, beat in enumerate(script.beats):
            b_copy = beat.model_copy(deep=True)
            kept = [pid for pid in b_copy.panel_ids if pid in valid_set]
            if not kept and valid_ids:
                fallback_id = self._scoped_fallback_panel(idx, total_beats, valid_ids, cleaned_beats)
                kept = [fallback_id]
                grounding_issues.append(
                    CriticIssue(
                        beat_id=b_copy.beat_id,
                        severity="warning",
                        category="grounding",
                        message=f"Replaced invalid panel_ids with scoped {fallback_id}.",
                    )
                )
            b_copy.panel_ids = kept
            cleaned_beats.append(b_copy)

        target_total_words = (
            max(400, int(round(script_config.target_duration_min * script_config.words_per_minute)))
            if script_config is not None and len(cleaned_beats) >= 6
            else None
        )

        issues_eval, flagged_indices, flagged_notes = self._evaluate_issues(
            cleaned_beats,
            all_pages=all_pages,
            page_order=page_order,
            beat_to_allowed=beat_to_allowed,
            target_total_words=target_total_words,
            bible=bible,
        )

        # Step 2: Closed-Loop Showrunner Pass + Auto-Repair (up to 2 passes if errors persist)
        if self.polish_seams and self.llm.has_text_credentials and len(cleaned_beats) >= 6:
            has_budget_overrun = any(i.category == "budget" for i in issues_eval)
            run_global = len(seam_indices) >= 2 or has_budget_overrun
            cleaned_beats = self._polish_multi_part_seams(
                cleaned_beats,
                bible,
                seam_indices=seam_indices,
                flagged_indices=flagged_indices,
                flagged_notes=flagged_notes,
                panel_text_lookup=panel_text_lookup,
                run_global_showrunner=run_global,
                target_total_words=target_total_words,
            )
            issues_eval, flagged_indices, flagged_notes = self._evaluate_issues(
                cleaned_beats,
                all_pages=all_pages,
                page_order=page_order,
                beat_to_allowed=beat_to_allowed,
                target_total_words=target_total_words,
                bible=bible,
            )
            remaining_errors = {i.beat_id for i in issues_eval if i.severity == "error"}
            if remaining_errors:
                retry_indices = {
                    idx for idx, b in enumerate(cleaned_beats) if b.beat_id in remaining_errors
                } | flagged_indices
                cleaned_beats = self._polish_multi_part_seams(
                    cleaned_beats,
                    bible,
                    seam_indices=[],
                    flagged_indices=retry_indices,
                    flagged_notes=[i.message for i in issues_eval if i.severity == "error"],
                    panel_text_lookup=panel_text_lookup,
                    run_global_showrunner=False,
                    target_total_words=target_total_words,
                )
                issues_eval, _, _ = self._evaluate_issues(
                    cleaned_beats,
                    all_pages=all_pages,
                    page_order=page_order,
                    beat_to_allowed=beat_to_allowed,
                    target_total_words=target_total_words,
                    bible=bible,
                )

        # Step 3: Populate TTS pronunciation-normalized strings without mutating display subtitles
        for b in cleaned_beats:
            b.tts_text = apply_pronunciation_glossary(b.narration_vi, bible.pronunciation_map)

        all_issues = grounding_issues + issues_eval
        final_script = RecapScript(
            title_vi=script.title_vi,
            persona=script.persona,
            hook_strategy=script.hook_strategy,
            hook_summary=script.hook_summary,
            beats=cleaned_beats,
        )
        err_count = sum(1 for i in all_issues if i.severity == "error")
        warn_count = len(all_issues) - err_count
        report = CriticReport(
            passed=(err_count == 0),
            score=max(2.0, round(10.0 - 1.2 * err_count - 0.35 * warn_count, 2)),
            issues=all_issues,
            notes="Validated hard invariants, macro rhythm distribution, Showrunner flow, and TTS pronunciation.",
        )
        final_path.write_text(final_script.model_dump_json(indent=2), encoding="utf-8")
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return final_script, report
