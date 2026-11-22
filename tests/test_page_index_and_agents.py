"""Tests for Circuit Breaker, PageAnalyzer micro-batch fallback, PageIndex, 3-tier Scriptwriter, and Critic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.critic import CriticAgent
from src.agents.page_analyzer import PageAnalyzerAgent, PageBatchAnalysis
from src.agents.page_index import PageIndexBuilder
from src.agents.scriptwriter import OutlinePart, PartScriptResponse, ScriptOutline, ScriptwriterAgent
from src.config import LLMConfig, ScriptConfig
from src.core.telemetry import TelemetryTracker
from src.llm.client import LLMClient
from src.schemas.analysis import PageAnalysis, PanelAnalysis, StoryBible
from src.schemas.ingestion import BBox, ChapterInfo, JobManifest, PageInfo, PanelInfo
from src.schemas.script import RecapScript, ScriptBeat


def _make_dummy_manifest(num_pages: int, panels_per_page: int = 2) -> tuple[JobManifest, list[PageAnalysis]]:
    pages: list[PageInfo] = []
    analyses: list[PageAnalysis] = []
    for p_idx in range(1, num_pages + 1):
        pid = f"ch01_p{p_idx:03d}"
        pnls: list[PanelInfo] = []
        pa_pnls: list[PanelAnalysis] = []
        for k in range(1, panels_per_page + 1):
            panel_id = f"{pid}_P{k}"
            pnls.append(
                PanelInfo(
                    panel_id=panel_id,
                    chapter_id="ch01",
                    page_id=pid,
                    index_in_page=k,
                    global_index=(p_idx - 1) * panels_per_page + k,
                    bbox=BBox(x1=0, y1=0, x2=100, y2=100),
                    width=100,
                    height=100,
                    crop_path=f"{panel_id}.jpg",
                )
            )
            pa_pnls.append(
                PanelAnalysis(
                    panel_id=panel_id,
                    action_summary=f"Action at {panel_id}",
                    characters_present=["Oz"] if p_idx % 2 == 0 else ["Gilbert"],
                    importance=4,
                )
            )
        pages.append(
            PageInfo(
                page_id=pid,
                chapter_id="ch01",
                page_index=p_idx,
                source_path=f"{pid}.jpg",
                width=200,
                height=300,
                panels=pnls,
            )
        )
        analyses.append(
            PageAnalysis(
                page_id=pid,
                chapter_id="ch01",
                page_summary=f"Summary {pid}",
                panels=pa_pnls,
            )
        )
    manifest = JobManifest(
        job_id="test_job",
        input_path="in",
        chapters=[ChapterInfo(chapter_id="ch01", title="ch01", pages=pages)],
    )
    return manifest, analyses


def test_llm_circuit_breaker_trips_primary_model() -> None:
    cfg = LLMConfig(
        default_model="gpt-6-luna",
        fallback_model="gemini-3.8-flash-low",
        api_key="dummy",
        max_retries=0,
    )
    tel = TelemetryTracker("j", "gpt-6-luna", "v")
    client = LLMClient(cfg, telemetry=tel)

    mock_openai = MagicMock()
    ok_resp = MagicMock()
    ok_resp.choices = [MagicMock(message=MagicMock(content='{"title_vi":"T","hook_summary":"H","beats":[]}'))]
    ok_resp.usage = None

    def side_effect(*, model, **kwargs):
        if model == "gpt-6-luna":
            raise RuntimeError("500 upstream error")
        return ok_resp

    mock_openai.chat.completions.create.side_effect = side_effect
    client._text_client = mock_openai

    # Call 1: tries gpt-6-luna (fails), falls back to gemini-3.8-flash-low, trips circuit breaker
    client.complete_json("A1", "sys", "usr", RecapScript)
    assert "gpt-6-luna" in client._tripped_models

    # Call 2: skips gpt-6-luna immediately -> only 1 create() call instead of 2
    before = mock_openai.chat.completions.create.call_count
    client.complete_json("A2", "sys", "usr", RecapScript)
    assert mock_openai.chat.completions.create.call_count == before + 1


def test_page_analyzer_batch_partial_fallback(tmp_path: Path) -> None:
    manifest, _ = _make_dummy_manifest(2)
    for pg in manifest.chapters[0].pages:
        img_p = tmp_path / pg.source_path
        img_p.write_bytes(b"fake")
        pg.source_path = str(img_p)

    cfg = LLMConfig(api_key="dummy", vision_api_key="dummy")
    llm = LLMClient(cfg)

    def fake_complete(agent_name, system_prompt, user_prompt, schema_cls, image_paths=None, **kw):
        if schema_cls is PageBatchAnalysis:
            # Return only page 1; page 2 is omitted -> must trigger single-page fallback
            return PageBatchAnalysis(
                pages=[PageAnalysis(page_id="ch01_p001", chapter_id="ch01", page_summary="batch_p1", panels=[])]
            )
        return PageAnalysis(page_id="ch01_p002", chapter_id="ch01", page_summary="single_p2", panels=[])

    llm.complete_json = fake_complete  # type: ignore
    agent = PageAnalyzerAgent(llm)
    out = agent.run_all(manifest, tmp_path / "job", max_workers=1, batch_size=2)
    assert [p.page_summary for p in out] == ["batch_p1", "single_p2"]


def test_page_index_2_tier_and_3_tier(tmp_path: Path) -> None:
    llm = LLMClient(LLMConfig(offline_fallback=True))
    m_small, a_small = _make_dummy_manifest(45)
    tree2, bible2 = PageIndexBuilder(llm, pages_per_scene=20).build(m_small, a_small, tmp_path / "j2")
    assert tree2.tiers == 2
    assert tree2.total_scenes == 3
    assert (tmp_path / "j2" / "repo" / "MEMORY.md").exists()
    assert len(bible2.characters) >= 2

    # > 40 scenes triggers 3-tier Arc->Scene tree automatically
    m_big, a_big = _make_dummy_manifest(210)
    tree3, _ = PageIndexBuilder(llm, pages_per_scene=5).build(m_big, a_big, tmp_path / "j3")
    assert tree3.tiers == 3
    assert len(tree3.arcs) >= 4


def test_scriptwriter_3_tier_routing_and_critic_scoped_fallback(tmp_path: Path) -> None:
    manifest, analyses = _make_dummy_manifest(40)
    llm_offline = LLMClient(LLMConfig(offline_fallback=True))
    tree, bible = PageIndexBuilder(llm_offline, pages_per_scene=20).build(manifest, analyses, tmp_path / "j")

    # Mock live LLM to verify Full-Context Multi-Part + lookup turn
    llm = LLMClient(LLMConfig(api_key="dummy"))
    calls: list[str] = []

    def fake_complete(agent_name, system_prompt, user_prompt, schema_cls, image_paths=None, **kw):
        calls.append(agent_name)
        if schema_cls is ScriptOutline:
            return ScriptOutline(
                title_vi="Test Vol",
                hook_summary="Hook",
                parts=[
                    OutlinePart(part_id="part_01", title="P1", scene_ids=["scene_001"]),
                    OutlinePart(part_id="part_02", title="P2", scene_ids=["scene_002"]),
                ],
            )
        return PartScriptResponse(
            beats=[
                ScriptBeat(
                    beat_id="b1",
                    narration_vi="Lời thoại test.",
                    panel_ids=["ch01_p001_P1"],
                )
            ]
        )

    llm.complete_json = fake_complete  # type: ignore
    writer = ScriptwriterAgent(llm, ScriptConfig(fast_path_max_pages=10, full_context_max_tokens=100000))
    script = writer.write_script(bible, analyses, "kich_tinh", tmp_path / "j", tree=tree)
    assert "OutlinePlannerAgent" in calls
    assert len(script.beats) == 2

    # Critic scoped fallback: beat 2 has invalid panel_id, should pick next after ch01_p001_P1 (ch01_p001_P2)
    script.beats[1].panel_ids = ["INVALID_ID"]
    critic = CriticAgent(llm, polish_seams=False)
    final_s, report = critic.review(script, manifest, bible, tmp_path / "j")
    assert final_s.beats[1].panel_ids == ["ch01_p001_P2"]
    assert any(i.category == "grounding" for i in report.issues)


def test_finish_reason_length_raises_and_critic_flags_gaps_and_cliches(tmp_path: Path) -> None:
    # 1) finish_reason == "length" must raise RuntimeError
    cfg = LLMConfig(default_model="claude-sonnet-4-6", fallback_model=None, api_key="dummy", max_retries=0)
    client = LLMClient(cfg)
    mock_openai = MagicMock()
    trunc_resp = MagicMock()
    trunc_resp.choices = [MagicMock(finish_reason="length", message=MagicMock(content='{"partial":'))]
    trunc_resp.usage = None
    mock_openai.chat.completions.create.return_value = trunc_resp
    client._text_client = mock_openai

    try:
        client.complete_json("Test", "sys", "usr", RecapScript)
        assert False, "Expected RuntimeError on finish_reason='length'"
    except RuntimeError as exc:
        assert "finish_reason='length'" in str(exc)

    # 2) Backmatter filtering + handoff tail helper
    manifest, analyses = _make_dummy_manifest(25)
    analyses[-2].page_summary = "Trang kết thúc chương 1"
    analyses[-1].page_summary = "Translator's Notes 1"
    llm_off = LLMClient(LLMConfig(offline_fallback=True))
    tree, bible = PageIndexBuilder(llm_off, pages_per_scene=20).build(manifest, analyses, tmp_path / "jgap")
    assert "ch01_p025" in tree.skipped_pages
    tail_txt, tail_pids = tree.get_scene_tail_text("scene_001", "ch01_p015")
    assert "ch01_p016" in tail_txt
    assert "ch01_p020_P2" in tail_pids

    # 3) Critic detects big page jump (>10 pages) and AI cliché
    script = RecapScript(
        title_vi="Gap Test",
        beats=[
            ScriptBeat(beat_id="b001", narration_vi="Mở đầu bình thường.", panel_ids=["ch01_p002_P1"]),
            ScriptBeat(beat_id="b002", narration_vi="Bỗng dưng định mệnh nghiệt ngã ập tới.", panel_ids=["ch01_p020_P1"]),
        ],
    )
    critic = CriticAgent(llm_off, polish_seams=False)
    _, rep = critic.review(script, manifest, bible, tmp_path / "jgap")
    cats = {i.category for i in rep.issues}
    assert "coverage_gap" in cats
    assert "style" in cats


def test_dummy_cache_rejection_and_cast_sheet_builder(tmp_path: Path) -> None:
    import cv2
    import numpy as np
    from src.agents.page_analyzer import (
        SeedCharacter,
        _build_cast_sheet,
        _is_dummy_page_analysis,
    )
    from src.schemas.ingestion import FocalPoint

    manifest, _ = _make_dummy_manifest(2)
    pnl = manifest.chapters[0].pages[0].panels[0]
    crop_file = tmp_path / "p1.jpg"
    cv2.imwrite(str(crop_file), np.full((240, 240, 3), 180, dtype=np.uint8))
    pnl.crop_path = str(crop_file)
    pnl.width = 240
    pnl.height = 240
    pnl.focal_points = [
        FocalPoint(
            kind="face",
            bbox=BBox(x1=60, y1=40, x2=160, y2=150),
            norm_cx=0.46,
            norm_cy=0.40,
            norm_w=0.42,
            norm_h=0.46,
            confidence=0.92,
        )
    ]

    sheet = _build_cast_sheet(
        [SeedCharacter(canonical_name="Oz Vessalius", evidence_quote="Oz!", best_panel_id=pnl.panel_id)],
        manifest,
        tmp_path / "repo" / "cast_sheet.jpg",
    )
    assert sheet is not None and sheet.exists()

    dummy_pa = PageAnalysis(
        page_id="ch01_p001",
        chapter_id="ch01",
        page_summary="Trang 16 của ch01 gồm 6 khung hình.",
        panels=[PanelAnalysis(panel_id="ch01_p001_P1", action_summary="Khung 1: diễn biến cảnh truyện.")],
    )
    assert _is_dummy_page_analysis(dummy_pa)
    real_pa = PageAnalysis(
        page_id="ch01_p001",
        chapter_id="ch01",
        page_summary="Oz nói chuyện với Gilbert trong hầm mộ.",
        panels=[PanelAnalysis(panel_id="ch01_p001_P1", action_summary="Oz cầm đồng hồ quả quýt.")],
    )
    assert not _is_dummy_page_analysis(real_pa)


def test_llm_client_passes_provider_for_text_only_not_vision(tmp_path: Path) -> None:
    cfg = LLMConfig(
        provider="aixingialaire",
        vision_provider=None,
        default_model="gpt-6-luna",
        fallback_model=None,
        vision_model="gemini-3.8-flash-medium-vision",
        api_key="dummy",
        max_retries=0,
    )
    client = LLMClient(cfg)
    captured_kwargs: list[dict] = []

    mock_openai = MagicMock()
    ok_resp = MagicMock()
    ok_resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content='{"title_vi":"T","hook_summary":"H","beats":[]}'))]
    ok_resp.usage = None

    def side_effect(**kwargs):
        captured_kwargs.append(kwargs)
        return ok_resp

    mock_openai.chat.completions.create.side_effect = side_effect
    client._text_client = mock_openai
    client._vision_client = mock_openai

    client.complete_json("TextAgent", "sys", "usr", RecapScript)
    assert captured_kwargs[-1].get("extra_body") == {"provider": "aixingialaire"}

    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    client.complete_json("VisionAgent", "sys", "usr", RecapScript, image_paths=[img])
    assert "extra_body" not in captured_kwargs[-1]


def test_frontmatter_and_blank_page_filtering(tmp_path: Path) -> None:
    import numpy as np
    import cv2
    from src.agents.page_analyzer import _is_blank_or_solid_page

    # Blank white image triggers deterministic filler detection without Vision hallucination
    white = tmp_path / "blank.jpg"
    cv2.imwrite(str(white), np.full((300, 200, 3), 255, dtype=np.uint8))
    assert _is_blank_or_solid_page(white)

    manifest, analyses = _make_dummy_manifest(12, panels_per_page=1)
    analyses[0].page_summary = "[PHỤ LỤC] Trang bìa tập 1"
    analyses[1].page_summary = "[PHỤ LỤC] Trang trắng không có nội dung"
    analyses[2].page_summary = "[PHỤ LỤC] Trang mục lục Retrace I"
    analyses[3].panels[0].dialogue_summary = '"Bắt đầu câu chuyện!"'
    llm_off = LLMClient(LLMConfig(offline_fallback=True))
    tree, _ = PageIndexBuilder(llm_off, pages_per_scene=20).build(manifest, analyses, tmp_path / "j_fm")
    assert set(["ch01_p001", "ch01_p002", "ch01_p003"]).issubset(set(tree.skipped_pages))
    assert tree.scenes[0].page_ids[0] == "ch01_p004"


def test_duplicate_spread_pages_skipped(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    manifest, analyses = _make_dummy_manifest(8, panels_per_page=2)
    pages = manifest.chapters[0].pages
    # Build real images for pages so _detect_duplicate_spread_pages compares pixels
    for i, pg in enumerate(pages):
        img_p = tmp_path / f"pg_{i+1}.jpg"
        if i == 1:
            # Landscape double-page spread: left half value ~40, right half value ~210
            spread = np.zeros((150, 200), dtype=np.uint8)
            spread[:, :100] = 40
            spread[:, 100:] = 210
            cv2.imwrite(str(img_p), spread)
            pg.width, pg.height = 200, 150
        elif i == 2:
            # Split RTL first half (matches right half of spread = 210)
            cv2.imwrite(str(img_p), np.full((150, 100), 210, dtype=np.uint8))
            pg.width, pg.height = 100, 150
        elif i == 3:
            # Split RTL second half (matches left half of spread = 40)
            cv2.imwrite(str(img_p), np.full((150, 100), 40, dtype=np.uint8))
            pg.width, pg.height = 100, 150
        else:
            cv2.imwrite(str(img_p), np.full((150, 100), (i * 25) % 255, dtype=np.uint8))
            pg.width, pg.height = 100, 150
        pg.source_path = str(img_p)
        analyses[i].panels[0].dialogue_summary = '"Thoại truyện"'

    llm_off = LLMClient(LLMConfig(offline_fallback=True))
    tree, _ = PageIndexBuilder(llm_off, pages_per_scene=20).build(manifest, analyses, tmp_path / "j_dup")
    assert "ch01_p003" in tree.skipped_pages
    assert "ch01_p004" in tree.skipped_pages
    assert "ch01_p002" in tree.scenes[0].page_ids


def test_critic_flags_duplication_and_backward_jump(tmp_path: Path) -> None:
    manifest, analyses = _make_dummy_manifest(10, panels_per_page=2)
    llm_off = LLMClient(LLMConfig(offline_fallback=True))
    _, bible = PageIndexBuilder(llm_off, pages_per_scene=20).build(manifest, analyses, tmp_path / "j_dup2")

    repeated = "Cậu bé hoảng loạn chạy vào rừng tìm người bạn thân đã biến mất từ lâu."
    script = RecapScript(
        title_vi="Dup & Back Test",
        beats=[
            ScriptBeat(beat_id="b001", narration_vi=repeated, panel_ids=["ch01_p005_P1"]),
            ScriptBeat(beat_id="b002", narration_vi=repeated, panel_ids=["ch01_p002_P1"]),
        ],
    )
    critic = CriticAgent(llm_off, polish_seams=False)
    _, rep = critic.review(script, manifest, bible, tmp_path / "j_dup2")
    cats = {i.category for i in rep.issues}
    assert "duplication" in cats
    assert "panel_sync" in cats


def test_no_series_specific_overfitting_in_codebase() -> None:
    import re

    forbidden = ("pandora", "vessalius", "b-rabbit", "gilbert", "sharon", "mochizuki", "baskerville", "alice", "oz")
    forbidden_substrings = (
        "ngã nhào khỏi tủ",
        "chính là sự tồn tại",
        "cô gái tóc đen",
        "chàng trai tóc sáng",
        "nền trắng và dòng chữ tên tác phẩm",
        "idx in (0, 2, 3, 4)",
        "blood_count",
    )
    for root_name in ("src", "configs"):
        for f in Path(root_name).rglob("*"):
            if not f.is_file() or f.suffix not in (".py", ".yaml", ".yml"):
                continue
            low = f.read_text(encoding="utf-8").lower()
            for kw in forbidden:
                assert not re.search(rf"\b{re.escape(kw)}\b", low), f"Series keyword {kw!r} found in {f}"
            for sub in forbidden_substrings:
                assert sub not in low, f"Hardcoded substring {sub!r} found in {f}"


def test_mood_contract_and_bidirectional_spread_and_early_skip(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    # 1) Free-form mood coercion into 7 canonical moods
    assert ScriptBeat(beat_id="b1", narration_vi="x", mood="explosive").mood == "action"
    assert ScriptBeat(beat_id="b2", narration_vi="x", mood="horrific").mood == "tense"
    assert ScriptBeat(beat_id="b3", narration_vi="x", mood="desperate").mood == "tragic"
    assert ScriptBeat(beat_id="b4", narration_vi="x", mood="playful").mood == "comedic"

    # 2) Bidirectional spread detection: [P1, P2, Spread] order
    manifest, _ = _make_dummy_manifest(6, panels_per_page=1)
    pages = manifest.chapters[0].pages
    for i, pg in enumerate(pages):
        img_p = tmp_path / f"bi_{i+1}.jpg"
        if i == 1:
            cv2.imwrite(str(img_p), np.full((150, 100), 210, dtype=np.uint8))
            pg.width, pg.height = 100, 150
        elif i == 2:
            cv2.imwrite(str(img_p), np.full((150, 100), 40, dtype=np.uint8))
            pg.width, pg.height = 100, 150
        elif i == 3:
            spread = np.zeros((150, 200), dtype=np.uint8)
            spread[:, :100] = 40
            spread[:, 100:] = 210
            cv2.imwrite(str(img_p), spread)
            pg.width, pg.height = 200, 150
        else:
            cv2.imwrite(str(img_p), np.full((150, 100), (i * 30 + 20) % 255, dtype=np.uint8))
            pg.width, pg.height = 100, 150
        pg.source_path = str(img_p)

    # PageAnalyzerAgent.run_all skips P1 & P2 deterministically before Vision calls
    llm = LLMClient(LLMConfig(offline_fallback=True))
    out = PageAnalyzerAgent(llm).run_all(manifest, tmp_path / "j_bi", max_workers=1)
    assert "[PHỤ LỤC] Trang đơn cắt trùng" in out[1].page_summary
    assert "[PHỤ LỤC] Trang đơn cắt trùng" in out[2].page_summary
    assert "[PHỤ LỤC]" not in out[3].page_summary


def test_full_alias_resolution_compact_md_and_closed_loop_critic(tmp_path: Path) -> None:
    from src.agents.critic import _SeamPolishResponse, _SeamRewrite
    from src.agents.page_index import _AliasResolutionResponse, _AliasRule, format_pages_compact_md

    manifest, analyses = _make_dummy_manifest(12, panels_per_page=2)
    analyses[0].verbatim_quote_vi = "Không đời nào ta bỏ cuộc!"
    analyses[0].visual_punch = "Ánh mắt sắc lạnh dưới mũ trùm"
    analyses[0].page_summary = "[Chưa rõ tên: người mặc áo choàng] bước vào."
    analyses[0].panels[0].characters_present = ["[Chưa rõ tên: người mặc áo choàng]"]
    analyses[0].panels[0].action_summary = "[Chưa rõ tên: người mặc áo choàng] rút kiếm."
    analyses[6].page_summary = "Ken cười lớn."
    analyses[6].panels[0].characters_present = ["Ken"]

    compact = format_pages_compact_md(analyses[:1])
    assert "⚡Thoại:" in compact and "👁️Hình:" in compact

    llm = LLMClient(LLMConfig(api_key="dummy"))

    def fake_alias(agent_name, system_prompt, user_prompt, schema_cls, **kw):
        if schema_cls is _AliasResolutionResponse:
            return _AliasResolutionResponse(
                rules=[
                    _AliasRule(
                        raw_name="[Chưa rõ tên: người mặc áo choàng]",
                        canonical_name="Arthur",
                        page_ids=["ch01_p001"],
                    )
                ]
            )
        return schema_cls()

    llm.complete_json = fake_alias  # type: ignore
    resolved = PageIndexBuilder(llm)._resolve_character_aliases(analyses, tmp_path / "j_al")
    assert "Arthur" in resolved[0].page_summary
    assert "Arthur" in resolved[0].panels[0].action_summary

    # Closed-Loop Critic Auto-Repair resolves clichés and locks panel_ids
    _, bible = PageIndexBuilder(LLMClient(LLMConfig(offline_fallback=True))).build(
        manifest, analyses, tmp_path / "j_cr"
    )
    bible.pronunciation_map = {"Arthur": "A thơ"}

    distinct_texts = [
        "Nhân vật chính mở cánh cửa gỗ cũ kỹ bước vào đại sảnh.",
        "Cậu phát hiện chiếc đồng hồ quả quýt phát sáng trên bàn đá.",
        "Chuyển sang phần mới, bánh răng định mệnh quay cuồng.",
        "Cánh cửa mật thất đóng sầm khiến cả nhóm bị kẹt dưới hầm.",
        "Đội hiệp sĩ hoàng gia lập tức bao vây khu rừng phía bắc.",
        "Cả hai vội vã trèo qua tường thành để trốn khỏi cuộc truy đuổi.",
    ]
    beats = [
        ScriptBeat(beat_id=f"b{i+1:03d}", narration_vi=distinct_texts[i], panel_ids=[f"ch01_p{i+1:03d}_P1"])
        for i in range(6)
    ]

    llm_c = LLMClient(LLMConfig(api_key="dummy"))

    def fake_polish(agent_name, system_prompt, user_prompt, schema_cls, **kw):
        return _SeamPolishResponse(
            rewrites=[_SeamRewrite(beat_id="b003", narration_vi="Arthur lập tức lao lên phá vỡ vòng vây.")]
        )

    llm_c.complete_json = fake_polish  # type: ignore
    critic = CriticAgent(llm_c, polish_seams=True)
    final_s, rep = critic.review(RecapScript(title_vi="T", beats=beats), manifest, bible, tmp_path / "j_cr")
    assert rep.passed is True
    assert not any(i.category == "style" for i in rep.issues)
    assert final_s.beats[2].panel_ids == ["ch01_p003_P1"]
    assert "A thơ" in final_s.beats[2].tts_text
