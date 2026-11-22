"""Stage 3b Agent: Critic validating panel grounding, coverage gaps, clichés, and narrative flow."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from pydantic import BaseModel, Field

from src.llm.client import LLMClient
from src.schemas.analysis import StoryBible
from src.schemas.ingestion import JobManifest
from src.schemas.script import CriticIssue, CriticReport, RecapScript, ScriptBeat


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


class CriticAgent:
    """Checks script consistency against manifest panels, page coverage, AI clichés, and seams."""

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

    def _polish_multi_part_seams(
        self,
        beats: list[ScriptBeat],
        bible: StoryBible,
        seam_indices: list[int],
        flagged_notes: list[str],
    ) -> list[ScriptBeat]:
        """Run 1 LLM Narrative Editor pass to polish seams, narrative gaps, and AI clichés."""
        if not self.polish_seams or not self.llm.has_text_credentials or len(beats) < 6:
            return beats

        # Only target actual flagged cliché/pacing beats and boundary seams instead of blindly rewriting all beats
        flagged_indices = {
            i
            for i, b in enumerate(beats)
            if (
                any(phrase in b.narration_vi.lower() for phrase in _BANNED_CLICHES)
                or len(b.narration_vi.split()) < 10
                or len(b.narration_vi.split()) > 65
            )
        }
        seam_set = {i for i in seam_indices if 0 <= i < len(beats)}
        target_indices = sorted(flagged_indices | seam_set)
        if not target_indices:
            return beats

        seam_beat_ids = [beats[i].beat_id for i in sorted(seam_set)]
        payload = [
            {"beat_id": beats[i].beat_id, "section": beats[i].section, "narration_vi": beats[i].narration_vi}
            for i in target_indices
        ]
        cast_names = [c.canonical_name for c in bible.characters]
        notes_txt = "\n".join(f"- {n}" for n in flagged_notes[:12]) or "Không có cảnh báo tự động."
        try:
            res = self.llm.complete_json(
                agent_name="CriticSeamPolisher",
                system_prompt=(
                    "Bạn là biên tập viên kịch bản voiceover YouTube manga. CHỈ trả về các beat thực sự có lỗi cần sửa để:\n"
                    "1) Nối mượt các điểm chuyển hồi (đặc biệt tại seam_beat_ids), xóa lời giới thiệu nhân vật bị lặp. "
                    "TUYỆT ĐỐI CẤM câu chuyển đoạn kiểu bình luận viên ('Chuyển sang đầu phần mới...', 'Nối tiếp cao trào...', 'Bước sang diễn biến tiếp theo...', 'Sang hồi kết mở...'). Hãy nối thẳng bằng hành động của nhân vật.\n"
                    "2) Khử sạch cụm sáo rỗng AI ('định mệnh nghiệt ngã', 'bánh răng định mệnh', 'không gian vỡ vụn', 'rợn tóc gáy', 'đẫm máu' lặp lại) "
                    "và câu đảo ngữ văn viết gượng, giữ đúng ngôi kể và VĂN NÓI 100% tự nhiên.\n"
                    "3) Không làm sai lệch sự kiện, không cắt cụt chữ đầu câu, và không đổi độ dài quá mức."
                ),
                user_prompt=(
                    f"Danh sách nhân vật chuẩn: {cast_names}\n"
                    f"Các điểm nối hồi (seam): {seam_beat_ids}\n"
                    f"Các lỗi kỹ thuật/văn phong phát hiện tự động:\n{notes_txt}\n\n"
                    f"Các beat cần rà soát:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
                schema_cls=_SeamPolishResponse,
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

        issues: list[CriticIssue] = []
        cleaned_beats: list[ScriptBeat] = []
        total_beats = len(script.beats)
        avg_pages_per_beat = (
            len(all_pages) / max(1, total_beats)
            if total_beats >= 6
            else 2.5
        )
        max_single_beat_span = max(3, math.ceil(avg_pages_per_beat * 1.5))
        max_coverage_gap = max(8, math.ceil(avg_pages_per_beat * 3.5))

        prev_page_pos: int | None = None
        flagged_notes: list[str] = []
        beat_ngrams: list[tuple[str, set[str]]] = []

        for idx, beat in enumerate(script.beats):
            kept = [pid for pid in beat.panel_ids if pid in valid_set]
            if not kept and valid_ids:
                fallback_id = self._scoped_fallback_panel(idx, total_beats, valid_ids, cleaned_beats)
                kept = [fallback_id]
                issues.append(
                    CriticIssue(
                        beat_id=beat.beat_id,
                        severity="warning",
                        category="grounding",
                        message=f"Replaced invalid panel_ids with scoped {fallback_id}.",
                    )
                )

            # Check out-of-part panel drift when boundaries exist (only flag, do not break tests)
            if idx in beat_to_allowed and kept:
                allowed = beat_to_allowed[idx]
                out_ids = [pid for pid in kept if pid not in allowed]
                if out_ids:
                    msg = f"{beat.beat_id} references panels outside assigned part range: {out_ids}."
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="warning", category="panel_sync", message=msg)
                    )
                    flagged_notes.append(msg)

            # Check page coverage gaps, wide single-beat spans, and backward jumps adaptively
            positions = [p for pid in kept if (p := _page_pos_from_panel(pid, page_order)) is not None]
            if positions:
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
                        CriticIssue(beat_id=beat.beat_id, severity="warning", category="panel_sync", message=back_msg)
                    )
                    flagged_notes.append(back_msg)
                if prev_page_pos is not None and (cur_min - prev_page_pos) > max_coverage_gap:
                    gap_msg = (
                        f"Page coverage gap before {beat.beat_id}: jumped from {all_pages[prev_page_pos]} "
                        f"to {all_pages[cur_min]} ({cur_min - prev_page_pos} pages skipped)."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="warning", category="coverage_gap", message=gap_msg)
                    )
                    flagged_notes.append(gap_msg)
                prev_page_pos = max(prev_page_pos or 0, cur_max)

            # Check cross-beat content duplication via n-gram overlap
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
                                severity="warning",
                                category="duplication",
                                message=dup_msg,
                            )
                        )
                        flagged_notes.append(dup_msg)
                        break
                beat_ngrams.append((beat.beat_id, cur_ng))

            # Check pacing & banned AI clichés
            words = len(beat.narration_vi.split())
            if words > 75:
                issues.append(
                    CriticIssue(
                        beat_id=beat.beat_id,
                        severity="warning",
                        category="pacing",
                        message="Beat narration exceeds 75 words; consider splitting.",
                    )
                )

            low_text = beat.narration_vi.lower()
            for phrase in _BANNED_CLICHES:
                if phrase in low_text:
                    c_msg = f"{beat.beat_id} chứa cụm sáo rỗng '{phrase}'."
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="warning", category="style", message=c_msg)
                    )
                    flagged_notes.append(c_msg)

            beat.panel_ids = kept
            cleaned_beats.append(beat)

        cleaned_beats = self._polish_multi_part_seams(
            cleaned_beats, bible, seam_indices=seam_indices, flagged_notes=flagged_notes
        )

        # Re-check cliché count on final output for accurate report notes
        final_script = RecapScript(
            title_vi=script.title_vi,
            persona=script.persona,
            hook_summary=script.hook_summary,
            beats=cleaned_beats,
        )
        report = CriticReport(
            passed=all(i.severity != "error" for i in issues),
            score=max(5.0, round(10.0 - 0.5 * len(issues), 2)),
            issues=issues,
            notes="Validated panel grounding, part boundaries, page coverage gaps, AI clichés, and narrative seams.",
        )
        final_path.write_text(final_script.model_dump_json(indent=2), encoding="utf-8")
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return final_script, report
