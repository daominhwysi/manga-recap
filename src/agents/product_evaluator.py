"""Stage 7: Holistic Script + Visual Product Evaluator for Coding-Agent Closed Loop."""

from __future__ import annotations

import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.agents.critic import _BANNED_CLICHES, _page_pos_from_panel
from src.agents.page_analyzer import _draw_utf8_label
from src.agents.page_index import PageIndexTree
from src.config import ScriptConfig
from src.llm.client import LLMClient
from src.schemas.analysis import PageAnalysis, StoryBible
from src.schemas.ingestion import JobManifest, PanelInfo
from src.schemas.script import CriticReport, RecapScript
from src.schemas.timeline import VideoTimeline

_STOPWORDS_VI = {
    "và",
    "của",
    "là",
    "trong",
    "với",
    "một",
    "những",
    "các",
    "đã",
    "đang",
    "sẽ",
    "không",
    "này",
    "đó",
    "khi",
    "cho",
    "từ",
    "đến",
    "trên",
    "dưới",
    "lại",
    "như",
    "nhưng",
    "rằng",
    "thì",
    "vào",
    "ra",
    "bị",
    "được",
    "cũng",
    "chỉ",
    "nếu",
    "vì",
}


class ScriptHolisticMetrics(BaseModel):
    total_beats: int
    total_words: int
    target_words: int
    budget_ratio: float
    avg_words_per_beat: float
    words_per_beat_std: float
    short_beats_pct: float
    long_beats_pct: float
    hook_strategy: str
    mood_distribution: dict[str, int] = Field(default_factory=dict)
    beat_type_distribution: dict[str, int] = Field(default_factory=dict)
    opening_bigram_diversity: float
    quote_grounding_rate: float
    visual_punch_grounding_rate: float
    alias_leak_beats: list[str] = Field(default_factory=list)
    cliche_beats: list[str] = Field(default_factory=list)
    repaired_beats_from_draft: int = 0


class VisualGroundingMetrics(BaseModel):
    total_manifest_pages: int
    skipped_pages_count: int
    story_pages_count: int
    covered_story_pages_count: int
    story_page_coverage_pct: float
    leaked_skipped_panels: list[str] = Field(default_factory=list)
    crammed_beats: list[str] = Field(default_factory=list)
    overloaded_single_panel_beats: list[str] = Field(default_factory=list)
    low_quality_or_extreme_ar_panels: list[str] = Field(default_factory=list)
    panels_with_focal_points_pct: float = 0.0
    total_timeline_shots: int = 0
    median_shot_sec: float = 0.0
    max_shot_sec: float = 0.0
    visual_freeze_shots: list[str] = Field(default_factory=list)
    visual_whiplash_shots: list[str] = Field(default_factory=list)
    camera_motion_distribution: dict[str, int] = Field(default_factory=dict)
    diptych_violations: list[str] = Field(default_factory=list)


class HolisticLLMReview(BaseModel):
    evaluated_by_llm: bool = False
    narrative_flow_score: float = 0.0
    visual_script_sync_score: float = 0.0
    character_consistency_score: float = 0.0
    visual_framing_score: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    beat_level_notes: list[str] = Field(default_factory=list)


class ActionableCodeFinding(BaseModel):
    severity: str
    category: str
    summary: str
    suspected_files: list[str] = Field(default_factory=list)
    recommended_fix: str
    minimal_rerun_cmd: str


class ProductEvalReport(BaseModel):
    job_id: str
    overall_score: float
    passed: bool
    script_metrics: ScriptHolisticMetrics
    visual_metrics: VisualGroundingMetrics
    llm_review: HolisticLLMReview = Field(default_factory=HolisticLLMReview)
    findings: list[ActionableCodeFinding] = Field(default_factory=list)
    contact_sheet_path: str = ""


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[^\W\d_]{2,}", text.lower(), flags=re.UNICODE)
    return {w for w in words if w not in _STOPWORDS_VI}


def _resolve_panel_img(job_dir: Path, crop_path: str) -> Path | None:
    p = Path(crop_path)
    if p.exists():
        return p
    rel = job_dir / p
    if rel.exists():
        return rel
    by_name = job_dir / "panels" / p.name
    return by_name if by_name.exists() else None


class ProductEvaluatorAgent:
    """Evaluates final script + visual assets as a unified product for coding-agent feedback."""

    def __init__(self, llm: LLMClient | None = None, script_config: ScriptConfig | None = None) -> None:
        self.llm = llm
        self.script_config = script_config or ScriptConfig()

    def build_visual_contact_sheet(
        self,
        job_dir: Path,
        manifest: JobManifest,
        script: RecapScript,
        out_path: Path,
        max_samples: int = 12,
    ) -> Path | None:
        """Sample up to 12 evenly spaced beat-panel cards into a labeled UTF-8 contact sheet."""
        if not script.beats:
            return None
        panel_map: dict[str, PanelInfo] = {
            pnl.panel_id: pnl
            for ch in manifest.chapters
            for pg in ch.pages
            for pnl in pg.panels
        }
        n = min(max_samples, len(script.beats))
        if n == len(script.beats):
            indices = list(range(len(script.beats)))
        else:
            indices = sorted({round(i * (len(script.beats) - 1) / max(1, n - 1)) for i in range(n)})

        cols = 3
        rows = max(1, math.ceil(len(indices) / cols))
        card_w, card_h = 420, 320
        img_h = 228
        canvas = np.full((rows * card_h, cols * card_w, 3), 22, dtype=np.uint8)

        for slot_i, b_idx in enumerate(indices):
            beat = script.beats[b_idx]
            r, c = divmod(slot_i, cols)
            x0, y0 = c * card_w, r * card_h
            cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + card_w - 4, y0 + card_h - 4), (48, 52, 62), 1)

            pid = beat.panel_ids[0] if beat.panel_ids else ""
            pinfo = panel_map.get(pid)
            img_path = _resolve_panel_img(job_dir, pinfo.crop_path) if pinfo else None
            if img_path is not None:
                img = cv2.imread(str(img_path))
                if img is not None:
                    ih, iw = img.shape[:2]
                    avail_w, avail_h = card_w - 16, img_h - 32
                    scale = min(avail_w / max(1, iw), avail_h / max(1, ih))
                    nw, nh = max(8, int(round(iw * scale))), max(8, int(round(ih * scale)))
                    resized = cv2.resize(
                        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                    )
                    ox = x0 + 8 + (avail_w - nw) // 2
                    oy = y0 + 28 + (avail_h - nh) // 2
                    canvas[oy : oy + nh, ox : ox + nw] = resized

            pages = sorted({re.sub(r"_P\d+$", "", p) for p in beat.panel_ids})
            pg_lbl = pages[0] if len(pages) == 1 else f"{pages[0]}..{pages[-1]}" if pages else "?"
            hdr = f"{beat.beat_id} [{beat.mood}] | {pg_lbl} | {len(beat.panel_ids)}p"
            canvas = _draw_utf8_label(canvas, hdr, (x0 + 10, y0 + 8), font_size=13, color_rgb=(255, 220, 120))

            wrapped = textwrap.wrap(beat.narration_vi.strip(), width=48)[:3]
            for line_i, line_txt in enumerate(wrapped):
                canvas = _draw_utf8_label(
                    canvas,
                    line_txt,
                    (x0 + 10, y0 + img_h + 6 + line_i * 22),
                    font_size=13,
                    color_rgb=(235, 238, 245),
                )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        return out_path

    def evaluate(self, job_dir: Path, use_llm: bool = True) -> ProductEvalReport:
        """Evaluate script + visual grounding/framing on a completed job directory."""
        manifest = JobManifest.model_validate_json((job_dir / "01_manifest.json").read_text(encoding="utf-8"))
        script = RecapScript.model_validate_json((job_dir / "03b_script_final.json").read_text(encoding="utf-8"))

        analyses: list[PageAnalysis] = []
        a_path = job_dir / "02a_all_pages.json"
        if a_path.exists():
            analyses = [PageAnalysis.model_validate(x) for x in json.loads(a_path.read_text(encoding="utf-8"))]
        analysis_map = {pa.page_id: pa for pa in analyses}

        draft_script: RecapScript | None = None
        draft_path = job_dir / "03a_script_draft.json"
        if draft_path.exists():
            draft_script = RecapScript.model_validate_json(draft_path.read_text(encoding="utf-8"))

        bible: StoryBible | None = None
        bible_path = job_dir / "02b_story_bible.json"
        if bible_path.exists():
            bible = StoryBible.model_validate_json(bible_path.read_text(encoding="utf-8"))

        tree: PageIndexTree | None = None
        tree_path = job_dir / "repo" / "tree_index.json"
        if tree_path.exists():
            tree = PageIndexTree.model_validate_json(tree_path.read_text(encoding="utf-8"))

        timeline: VideoTimeline | None = None
        tl_path = job_dir / "05_timeline.json"
        if tl_path.exists():
            timeline = VideoTimeline.model_validate_json(tl_path.read_text(encoding="utf-8"))

        critic_rep: CriticReport | None = None
        cr_path = job_dir / "03b_critic_report.json"
        if cr_path.exists():
            critic_rep = CriticReport.model_validate_json(cr_path.read_text(encoding="utf-8"))

        # Compute Script Metrics
        beats = script.beats
        word_counts = [len(b.narration_vi.split()) for b in beats]
        total_words = sum(word_counts)
        target_words = max(400, int(round(self.script_config.target_duration_min * self.script_config.words_per_minute)))
        avg_wc = round(total_words / max(1, len(beats)), 2)
        std_wc = (
            round(math.sqrt(sum((w - avg_wc) ** 2 for w in word_counts) / len(word_counts)), 2)
            if word_counts
            else 0.0
        )
        short_pct = round(100.0 * sum(1 for w in word_counts if w <= 28) / max(1, len(beats)), 1)
        long_pct = round(100.0 * sum(1 for w in word_counts if w >= 50) / max(1, len(beats)), 1)

        mood_dist = dict(Counter(b.mood for b in beats))
        type_dist = dict(Counter(b.beat_type for b in beats))
        openings = [" ".join(b.narration_vi.lower().split()[:2]) for b in beats if b.narration_vi.strip()]
        opening_div = round(len(set(openings)) / max(1, len(openings)), 3)

        alias_leaks: list[str] = []
        cliche_hits: list[str] = []
        for b in beats:
            low = b.narration_vi.lower()
            if "[chưa rõ" in low or "[nhân vật" in low or "chưa rõ tên" in low:
                alias_leaks.append(b.beat_id)
            if any(ph in low for ph in _BANNED_CLICHES):
                cliche_hits.append(b.beat_id)

        repaired_count = 0
        if draft_script is not None:
            draft_map = {b.beat_id: b.narration_vi for b in draft_script.beats}
            repaired_count = sum(1 for b in beats if draft_map.get(b.beat_id, b.narration_vi) != b.narration_vi)

        # Page -> Beats narration lookup (+/-1 beat window for natural transition)
        all_pages = [pg.page_id for ch in manifest.chapters for pg in ch.pages]
        page_order = {pid: idx for idx, pid in enumerate(all_pages)}
        skipped_set = set(tree.skipped_pages) if tree else set()
        story_pages = [pg_id for pg_id in all_pages if pg_id not in skipped_set]

        page_beat_idxs: dict[str, list[int]] = {}
        for idx, b in enumerate(beats):
            for pid in b.panel_ids:
                pg = re.sub(r"_P\d+$", "", pid)
                page_beat_idxs.setdefault(pg, []).append(idx)

        def _grounding_rate(attr: str) -> float:
            candidates = [
                pg_id
                for pg_id in story_pages
                if pg_id in analysis_map and getattr(analysis_map[pg_id], attr, "").strip()
            ]
            if not candidates:
                return 100.0
            hits = 0
            for pg_id in candidates:
                src_toks = _content_tokens(getattr(analysis_map[pg_id], attr))
                if not src_toks:
                    hits += 1
                    continue
                b_indices = set()
                for bi in page_beat_idxs.get(pg_id, []):
                    for d in (bi - 1, bi, bi + 1):
                        if 0 <= d < len(beats):
                            b_indices.add(d)
                joined_narration = " ".join(beats[i].narration_vi for i in b_indices)
                nar_toks = _content_tokens(joined_narration)
                overlap = len(src_toks & nar_toks)
                if overlap >= min(2, len(src_toks)) or (overlap / max(1, len(src_toks))) >= 0.35:
                    hits += 1
            return round(100.0 * hits / len(candidates), 1)

        quote_rate = _grounding_rate("verbatim_quote_vi")
        punch_rate = _grounding_rate("visual_punch")

        script_metrics = ScriptHolisticMetrics(
            total_beats=len(beats),
            total_words=total_words,
            target_words=target_words,
            budget_ratio=round(total_words / max(1, target_words), 2),
            avg_words_per_beat=avg_wc,
            words_per_beat_std=std_wc,
            short_beats_pct=short_pct,
            long_beats_pct=long_pct,
            hook_strategy=script.hook_strategy,
            mood_distribution=mood_dist,
            beat_type_distribution=type_dist,
            opening_bigram_diversity=opening_div,
            quote_grounding_rate=quote_rate,
            visual_punch_grounding_rate=punch_rate,
            alias_leak_beats=alias_leaks,
            cliche_beats=cliche_hits,
            repaired_beats_from_draft=repaired_count,
        )

        # Compute Visual Metrics
        panel_map: dict[str, PanelInfo] = {
            pnl.panel_id: pnl
            for ch in manifest.chapters
            for pg in ch.pages
            for pnl in pg.panels
        }
        covered_pages: set[str] = set()
        leaked_skipped: list[str] = []
        crammed: list[str] = []
        overloaded_single: list[str] = []
        ref_panel_ids: list[str] = []

        density = len(all_pages) / max(1, len(beats))
        max_span = max(3, min(8, round(density * 1.8)))

        for b in beats:
            positions = [p for pid in b.panel_ids if (p := _page_pos_from_panel(pid, page_order)) is not None]
            for pid in b.panel_ids:
                ref_panel_ids.append(pid)
                pg = re.sub(r"_P\d+$", "", pid)
                covered_pages.add(pg)
                if pg in skipped_set and pid not in leaked_skipped:
                    leaked_skipped.append(pid)
            if positions and (max(positions) - min(positions) + 1) > max_span:
                crammed.append(b.beat_id)
            if len(b.panel_ids) == 1 and len(b.narration_vi.split()) >= 40:
                overloaded_single.append(b.beat_id)

        covered_story = [pg for pg in story_pages if pg in covered_pages]
        coverage_pct = round(100.0 * len(covered_story) / max(1, len(story_pages)), 1)

        unique_ref_panels = list(dict.fromkeys(ref_panel_ids))
        bad_ar_panels: list[str] = []
        fp_count = 0
        for pid in unique_ref_panels:
            pinfo = panel_map.get(pid)
            if not pinfo:
                continue
            if pinfo.focal_points:
                fp_count += 1
            ar = pinfo.width / max(1, pinfo.height)
            if min(pinfo.width, pinfo.height) < 100 or ar < 0.2 or ar > 4.5:
                bad_ar_panels.append(f"{pid}({pinfo.width}x{pinfo.height})")
        fp_pct = round(100.0 * fp_count / max(1, len(unique_ref_panels)), 1)

        shot_durs: list[float] = []
        freeze_shots: list[str] = []
        whiplash_shots: list[str] = []
        motion_dist: dict[str, int] = {}
        diptych_violations: list[str] = []
        if timeline is not None and timeline.shots:
            motion_dist = dict(Counter(s.camera_move for s in timeline.shots))
            for s in timeline.shots:
                dur = max(0.0, (s.end_ms - s.start_ms) / 1000.0)
                shot_durs.append(dur)
                if dur > 5.5 and not s.secondary_panel_id:
                    freeze_shots.append(f"{s.shot_id}({s.beat_id}:{dur:.1f}s)")
                if dur < 0.85:
                    whiplash_shots.append(f"{s.shot_id}({s.beat_id}:{dur:.2f}s)")
                if s.secondary_panel_id:
                    pg1 = re.sub(r"_P\d+$", "", s.panel_id)
                    pg2 = re.sub(r"_P\d+$", "", s.secondary_panel_id)
                    if pg1 != pg2:
                        diptych_violations.append(f"{s.shot_id}({pg1}!={pg2})")
            sorted_durs = sorted(shot_durs)
            median_shot = round(sorted_durs[len(sorted_durs) // 2], 2)
            max_shot = round(max(sorted_durs), 2)
        else:
            median_shot = 0.0
            max_shot = 0.0

        visual_metrics = VisualGroundingMetrics(
            total_manifest_pages=len(all_pages),
            skipped_pages_count=len(skipped_set),
            story_pages_count=len(story_pages),
            covered_story_pages_count=len(covered_story),
            story_page_coverage_pct=coverage_pct,
            leaked_skipped_panels=leaked_skipped,
            crammed_beats=crammed,
            overloaded_single_panel_beats=overloaded_single,
            low_quality_or_extreme_ar_panels=bad_ar_panels,
            panels_with_focal_points_pct=fp_pct,
            total_timeline_shots=len(timeline.shots) if timeline else 0,
            median_shot_sec=median_shot,
            max_shot_sec=max_shot,
            visual_freeze_shots=freeze_shots,
            visual_whiplash_shots=whiplash_shots,
            camera_motion_distribution=motion_dist,
            diptych_violations=diptych_violations,
        )

        # Contact Sheet + Optional LLM Holistic Critique
        sheet_path = self.build_visual_contact_sheet(
            job_dir, manifest, script, job_dir / "07_visual_contact_sheet.jpg"
        )
        llm_review = HolisticLLMReview()
        if use_llm and self.llm is not None and (self.llm.has_text_credentials or self.llm.has_vision_credentials):
            llm_review = self._run_llm_holistic_review(
                script=script,
                analysis_map=analysis_map,
                bible=bible,
                sheet_path=sheet_path,
                script_metrics=script_metrics,
                visual_metrics=visual_metrics,
            )

        findings = self._build_findings(job_dir, script_metrics, visual_metrics, critic_rep, llm_review)

        # Score calculation (0-100)
        det_score = 100.0
        det_score -= 8.0 * len(leaked_skipped)
        det_score -= 5.0 * (len(alias_leaks) + len(cliche_hits))
        det_score -= 3.0 * len(crammed)
        det_score -= 2.0 * len(overloaded_single)
        det_score -= 2.0 * (len(freeze_shots) + len(diptych_violations))
        if script_metrics.budget_ratio > 1.25 or script_metrics.budget_ratio < 0.70:
            det_score -= 10.0
        if coverage_pct < 75.0:
            det_score -= 0.4 * (75.0 - coverage_pct)
        det_score = max(15.0, min(100.0, det_score))

        if llm_review.evaluated_by_llm:
            llm_avg_100 = 10.0 * (
                llm_review.narrative_flow_score
                + llm_review.visual_script_sync_score
                + llm_review.character_consistency_score
                + llm_review.visual_framing_score
            ) / 4.0
            overall_score = round(0.60 * det_score + 0.40 * llm_avg_100, 1)
        else:
            overall_score = round(det_score, 1)

        has_high = any(f.severity == "high" for f in findings)
        report = ProductEvalReport(
            job_id=manifest.job_id,
            overall_score=overall_score,
            passed=(not has_high and overall_score >= 75.0),
            script_metrics=script_metrics,
            visual_metrics=visual_metrics,
            llm_review=llm_review,
            findings=findings,
            contact_sheet_path=str(sheet_path) if sheet_path else "",
        )

        (job_dir / "07_eval_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (job_dir / "07_EVAL_REPORT.md").write_text(self._render_markdown_report(report), encoding="utf-8")
        return report

    def _run_llm_holistic_review(
        self,
        script: RecapScript,
        analysis_map: dict[str, PageAnalysis],
        bible: StoryBible | None,
        sheet_path: Path | None,
        script_metrics: ScriptHolisticMetrics,
        visual_metrics: VisualGroundingMetrics,
    ) -> HolisticLLMReview:
        rows: list[str] = []
        for b in script.beats:
            pgs = list(dict.fromkeys(re.sub(r"_P\d+$", "", pid) for pid in b.panel_ids))
            vis_snippets: list[str] = []
            for pg in pgs[:2]:
                pa = analysis_map.get(pg)
                if not pa:
                    continue
                q = f" | Quote: {pa.verbatim_quote_vi}" if pa.verbatim_quote_vi else ""
                vis_snippets.append(f"{pa.page_summary[:90]}{q}")
            rows.append(
                f"[{b.beat_id} | {b.section}/{b.mood} | {','.join(b.panel_ids[:3])}] "
                f"SRC=({' ; '.join(vis_snippets) or 'n/a'}) => SAY=\"{b.narration_vi}\""
            )

        cast_list = [c.canonical_name for c in bible.characters[:16]] if bible else []
        prompt = (
            f"Tác phẩm: {script.title_vi} (hook={script.hook_strategy})\n"
            f"Nhân vật chuẩn: {cast_list}\n"
            f"Thống kê nhanh: beats={script_metrics.total_beats}, words={script_metrics.total_words}/"
            f"{script_metrics.target_words}, coverage={visual_metrics.story_page_coverage_pct}%, "
            f"quote_grounding={script_metrics.quote_grounding_rate}%\n\n"
            "Bảng đối chiếu toàn bộ Kịch bản vs Dữ liệu Hình ảnh:\n"
            + "\n".join(rows)[:32000]
        )
        images = [sheet_path] if (sheet_path and sheet_path.exists() and self.llm and self.llm.has_vision_credentials) else None
        try:
            assert self.llm is not None
            res = self.llm.complete_json(
                agent_name="ProductEvaluatorAgent",
                system_prompt=(
                    "Bạn là Tổng Đạo Diễn & Biên Tập QA. Hãy chấm toàn diện sản phẩm Kịch bản + Hình ảnh như một thể thống nhất.\n"
                    "Quan sát cả mạch truyện chữ lẫn ảnh contact sheet đính kèm (nếu có) và trả về JSON:\n"
                    "- narrative_flow_score (0..10): độ cuốn hút hook, mượt mà chuyển hồi, nhịp ngắn-dài.\n"
                    "- visual_script_sync_score (0..10): lời thoại khớp hình từng beat, không spoil/lệch.\n"
                    "- character_consistency_score (0..10): nhất quán tên nhân vật, không lọt nhãn vô danh.\n"
                    "- visual_framing_score (0..10): bố cục khung cắt, không crop cụt/lệch hoặc đứng hình đơn điệu.\n"
                    "- strengths (2-4 điểm mạnh cụ thể), weaknesses (0-4 điểm yếu cụ thể), beat_level_notes (tối đa 6 ghi chú kèm beat_id nếu phát hiện lỗi)."
                ),
                user_prompt=prompt,
                schema_cls=HolisticLLMReview,
                image_paths=images,
            )
            res.evaluated_by_llm = True
            return res
        except Exception:
            return HolisticLLMReview(evaluated_by_llm=False)

    def _build_findings(
        self,
        job_dir: Path,
        sm: ScriptHolisticMetrics,
        vm: VisualGroundingMetrics,
        critic_rep: CriticReport | None,
        llm_rev: HolisticLLMReview,
    ) -> list[ActionableCodeFinding]:
        findings: list[ActionableCodeFinding] = []
        jid = job_dir.name

        if vm.leaked_skipped_panels:
            findings.append(
                ActionableCodeFinding(
                    severity="high",
                    category="front_backmatter_leak",
                    summary=f"{len(vm.leaked_skipped_panels)} panel thuộc trang phụ lục/trùng lọt vào kịch bản: {vm.leaked_skipped_panels[:5]}",
                    suspected_files=["src/agents/page_index.py", "src/agents/scriptwriter.py"],
                    recommended_fix="Kiểm tra hàm lọc trang phụ lục & `skipped_pages` khi dựng Scene/Part.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/repo/OUTLINE.json {job_dir}/repo/script {job_dir}/03* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if sm.budget_ratio > 1.25:
            findings.append(
                ActionableCodeFinding(
                    severity="high",
                    category="script_budget_overrun",
                    summary=f"Kịch bản dài {sm.total_words} từ, vượt >25% mục tiêu {sm.target_words} từ (tỷ lệ {sm.budget_ratio:.2f}x).",
                    suspected_files=["src/agents/scriptwriter.py", "src/agents/critic.py"],
                    recommended_fix="Ép ngân sách từ chặt hơn ở OutlinePlanner/Scriptwriter và đảm bảo Critic Round-2 nhắm vào các beat dài nhất.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/03b_* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if sm.alias_leak_beats:
            findings.append(
                ActionableCodeFinding(
                    severity="high",
                    category="character_alias_leak",
                    summary=f"Còn {len(sm.alias_leak_beats)} beat lọt nhãn chưa chuẩn hóa tên nhân vật: {sm.alias_leak_beats[:6]}",
                    suspected_files=["src/agents/page_index.py", "src/agents/scriptwriter.py"],
                    recommended_fix="Củng cố `_resolve_character_aliases` và cấm mẫu `[Chưa rõ tên` trong Scriptwriter.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/02b_story_bible.json {job_dir}/repo/tree_index.json {job_dir}/repo/notes {job_dir}/03* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if sm.cliche_beats:
            findings.append(
                ActionableCodeFinding(
                    severity="medium",
                    category="ai_cliche_style",
                    summary=f"{len(sm.cliche_beats)} beat còn chứa cụm cliché AI: {sm.cliche_beats[:6]}",
                    suspected_files=["src/agents/critic.py", "configs/personas.yaml"],
                    recommended_fix="Bổ sung cụm mẫu vào `_BANNED_CLICHES` và kiểm tra Showrunner Pass.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/03b_* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if vm.crammed_beats:
            findings.append(
                ActionableCodeFinding(
                    severity="medium",
                    category="visual_cramming",
                    summary=f"{len(vm.crammed_beats)} beat ôm quá nhiều trang ({vm.crammed_beats[:6]}), dễ kể lướt hoặc hụt hình.",
                    suspected_files=["src/agents/scriptwriter.py", "src/agents/page_index.py"],
                    recommended_fix="Tăng số `target_beats` cho Part có nhiều trang hoặc chia Scene nhỏ hơn ở SemanticSceneSegmenter.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/repo/OUTLINE.json {job_dir}/repo/script {job_dir}/03* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if vm.overloaded_single_panel_beats or vm.visual_freeze_shots:
            findings.append(
                ActionableCodeFinding(
                    severity="medium",
                    category="visual_freeze_risk",
                    summary=(
                        f"{len(vm.overloaded_single_panel_beats)} beat 1 panel nhưng >=40 từ "
                        f"({vm.overloaded_single_panel_beats[:4]}); {len(vm.visual_freeze_shots)} shot đứng hình >5.5s "
                        f"({vm.visual_freeze_shots[:4]})."
                    ),
                    suspected_files=["src/agents/choreographer.py", "src/agents/scriptwriter.py"],
                    recommended_fix="Ép tách sub-shot hoặc mượn thêm panel cùng trang trong Choreographer; giảm từ trên beat 1 panel.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/05_timeline.json {job_dir}/06* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if vm.visual_whiplash_shots:
            findings.append(
                ActionableCodeFinding(
                    severity="low",
                    category="visual_whiplash",
                    summary=f"{len(vm.visual_whiplash_shots)} shot ngắn <0.85s ({vm.visual_whiplash_shots[:4]}), có thể lướt hình quá nhanh.",
                    suspected_files=["src/agents/choreographer.py"],
                    recommended_fix="Giới hạn số slot tối đa trong 1 beat ngắn hoặc gộp 2 panel thành diptych.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/05_timeline.json {job_dir}/06* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if vm.diptych_violations:
            findings.append(
                ActionableCodeFinding(
                    severity="high",
                    category="diptych_cross_page",
                    summary=f"Phát hiện ghép đôi khác trang: {vm.diptych_violations[:4]}",
                    suspected_files=["src/agents/choreographer.py"],
                    recommended_fix="Kiểm tra điều kiện `p1.page_id == p2.page_id` trong `_is_diptych_candidate`.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/05_timeline.json {job_dir}/06* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if sm.quote_grounding_rate < 45.0:
            findings.append(
                ActionableCodeFinding(
                    severity="low",
                    category="underused_verbatim_quotes",
                    summary=f"Tỷ lệ đưa thoại đắt (`verbatim_quote_vi`) vào kịch bản mới đạt {sm.quote_grounding_rate}%.",
                    suspected_files=["src/agents/scriptwriter.py", "src/agents/page_index.py"],
                    recommended_fix="Nhấn mạnh lại tag `⚡Thoại` trong `render_compact_notes` và prompt của Scriptwriter.",
                    minimal_rerun_cmd=f"rm -rf {job_dir}/repo/script {job_dir}/03* && uv run manga-recap run ... --job-id {jid}",
                )
            )
        if critic_rep is not None and not critic_rep.passed:
            err_msgs = [i.message for i in critic_rep.issues if i.severity == "error"]
            if err_msgs and not any(f.category == "script_budget_overrun" for f in findings):
                findings.append(
                    ActionableCodeFinding(
                        severity="high",
                        category="critic_unresolved_errors",
                        summary=f"Stage 3b Critic vẫn còn lỗi chưa tự sửa xong: {err_msgs[:3]}",
                        suspected_files=["src/agents/critic.py"],
                        recommended_fix="Kiểm tra logic `remaining_errors` / `_polish_multi_part_seams` trong CriticAgent.",
                        minimal_rerun_cmd=f"rm -rf {job_dir}/03b_* && uv run manga-recap run ... --job-id {jid}",
                    )
                )
        for wn in llm_rev.weaknesses[:2]:
            findings.append(
                ActionableCodeFinding(
                    severity="medium",
                    category="llm_holistic_critique",
                    summary=wn,
                    suspected_files=["src/agents/scriptwriter.py", "src/agents/choreographer.py"],
                    recommended_fix="Rà soát trực tiếp các beat ghi chú trong `07_EVAL_REPORT.md` và tinh chỉnh prompt/cắt cảnh.",
                    minimal_rerun_cmd=f"uv run manga-recap evaluate {job_dir}",
                )
            )
        return findings

    @staticmethod
    def _render_markdown_report(r: ProductEvalReport) -> str:
        sm = r.script_metrics
        vm = r.visual_metrics
        status = "PASS ✅" if r.passed else "NEEDS ATTENTION ⚠️"
        lines = [
            f"# Holistic Script + Visual Evaluation Report — `{r.job_id}`",
            "",
            f"- **Overall Score**: **{r.overall_score}/100** ({status})",
            f"- **Visual Contact Sheet**: `{r.contact_sheet_path}`",
            "",
            "## 1. Script Quality & Narrative Cohesion",
            "| Metric | Value |",
            "|---|---|",
            f"| Hook Strategy | `{sm.hook_strategy}` |",
            f"| Total Beats / Words | {sm.total_beats} beats / {sm.total_words} từ (target ~{sm.target_words}, ratio **{sm.budget_ratio:.2f}x**) |",
            f"| Beat Length Distribution | avg={sm.avg_words_per_beat}w, std={sm.words_per_beat_std}w (short<=28w: {sm.short_beats_pct}%, long>=50w: {sm.long_beats_pct}%) |",
            f"| Mood Mix | `{json.dumps(sm.mood_distribution, ensure_ascii=False)}` |",
            f"| Beat Type Mix | `{json.dumps(sm.beat_type_distribution, ensure_ascii=False)}` |",
            f"| Opening Diversity (2-gram) | {sm.opening_bigram_diversity} |",
            f"| Verbatim Quote Grounding | {sm.quote_grounding_rate}% |",
            f"| Visual Punch Grounding | {sm.visual_punch_grounding_rate}% |",
            f"| Draft -> Final Edited Beats | {sm.repaired_beats_from_draft} beats polished |",
            f"| Alias Leaks / AI Clichés | {len(sm.alias_leak_beats)} alias / {len(sm.cliche_beats)} cliché |",
            "",
            "## 2. Visual Grounding, Coverage & Framing",
            "| Metric | Value |",
            "|---|---|",
            f"| Story Page Coverage | {vm.covered_story_pages_count}/{vm.story_pages_count} story pages (**{vm.story_page_coverage_pct}%**, skipped={vm.skipped_pages_count}/{vm.total_manifest_pages}) |",
            f"| Skipped-Page Leaks | {len(vm.leaked_skipped_panels)} ({vm.leaked_skipped_panels[:5]}) |",
            f"| Crammed Beats / Single-Panel>=40w | {len(vm.crammed_beats)} crammed / {len(vm.overloaded_single_panel_beats)} single-panel heavy |",
            f"| Panels with Focal Points | {vm.panels_with_focal_points_pct}% (bad AR/tiny: {len(vm.low_quality_or_extreme_ar_panels)}) |",
            f"| Timeline Shots (Stage 5) | total={vm.total_timeline_shots}, median={vm.median_shot_sec}s, max={vm.max_shot_sec}s |",
            f"| Freeze (>5.5s) / Whiplash (<0.85s) | {len(vm.visual_freeze_shots)} freeze / {len(vm.visual_whiplash_shots)} whiplash |",
            f"| Camera Motion Mix | `{json.dumps(vm.camera_motion_distribution, ensure_ascii=False)}` |",
            "",
        ]
        if r.llm_review.evaluated_by_llm:
            lr = r.llm_review
            lines.extend([
                "## 3. Unified AI Review (Script + Contact Sheet)",
                f"- **Narrative Flow**: {lr.narrative_flow_score}/10 | **Visual-Script Sync**: {lr.visual_script_sync_score}/10 | "
                f"**Character Consistency**: {lr.character_consistency_score}/10 | **Visual Framing**: {lr.visual_framing_score}/10",
                "- **Strengths**:",
                *[f"  - {s}" for s in lr.strengths],
                "- **Weaknesses**:",
                *[f"  - {w}" for w in lr.weaknesses],
                "- **Beat Notes**:",
                *[f"  - {n}" for n in lr.beat_level_notes],
                "",
            ])
        lines.append("## 4. Actionable Findings for Coding Agent")
        if not r.findings:
            lines.append("- No critical or medium defects detected. Product is clean.")
        else:
            for idx, f in enumerate(r.findings, start=1):
                lines.extend([
                    f"{idx}. **[{f.severity.upper()}] [{f.category}]** {f.summary}",
                    f"   - Suspected Files: `{', '.join(f.suspected_files)}`",
                    f"   - Recommended Fix: {f.recommended_fix}",
                    f"   - Minimal Re-run: `{f.minimal_rerun_cmd}`",
                ])
        return "\n".join(lines) + "\n"
