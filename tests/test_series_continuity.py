"""Unit & integration tests for Multi-Volume Series Continuity Engine (Portrait Vault, 3-Tier Memory, Critic Gate, Music Sync, Compiler & Orchestrator)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from src.agents.critic import CriticAgent
from src.agents.page_analyzer import SeedCharacter, _build_cast_sheet
from src.agents.page_index import _compute_character_reveal_pages
from src.audio.music_director import MusicCue, select_tracks_for_cues
from src.audio.music_library import MusicCatalog, MusicTrack
from src.config import AppConfig, LLMConfig
from src.llm.client import LLMClient
from src.schemas.analysis import CharacterProfile, PageAnalysis, PanelAnalysis, StoryBible
from src.schemas.ingestion import BBox, ChapterInfo, FocalPoint, JobManifest, PageInfo, PanelInfo
from src.schemas.script import RecapScript, ScriptBeat
from src.schemas.series import (
    SeriesCharacter,
    SeriesContext,
    SeriesGlossary,
    SeriesState,
    VolumeHandoff,
    VolumeLedgerEntry,
)
from src.series.compiler import (
    compile_series_videos,
    merge_music_credits,
    merge_srt_files,
    parse_volume_range,
    write_youtube_chapters,
)
from src.series.memory import SeriesStore
from src.series.orchestrator import SeriesOrchestrator, discover_series_volumes


def _make_panel_image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((320, 240, 3), color, dtype=np.uint8)
    cv2.circle(img, (120, 110), 42, (235, 225, 215), -1)
    cv2.imwrite(str(path), img)
    return path


def _make_vol_manifest(tmp_path: Path, job_id: str, color: tuple[int, int, int]) -> JobManifest:
    crop = _make_panel_image(tmp_path / job_id / "panels" / "ch01_p001_P1.jpg", color)
    pnl = PanelInfo(
        panel_id="ch01_p001_P1",
        page_id="ch01_p001",
        chapter_id="ch01",
        index_in_page=0,
        global_index=0,
        bbox=BBox(x1=0, y1=0, x2=240, y2=320),
        width=240,
        height=320,
        crop_path=str(crop),
        focal_points=[
            FocalPoint(
                kind="face",
                confidence=0.95,
                bbox=BBox(x1=75, y1=65, x2=165, y2=155),
                norm_cx=0.5,
                norm_cy=0.35,
                norm_w=0.37,
                norm_h=0.28,
            )
        ],
    )
    pg = PageInfo(
        page_id="ch01_p001",
        chapter_id="ch01",
        page_index=0,
        source_path=str(crop),
        width=240,
        height=320,
        panels=[pnl],
    )
    return JobManifest(
        job_id=job_id,
        input_path=str(tmp_path / job_id),
        reading_order="rtl",
        chapters=[ChapterInfo(chapter_id="ch01", title="Ch 1", pages=[pg])],
    )


def test_portrait_vault_persists_faces_across_volumes_and_prioritizes_active_cast(tmp_path: Path) -> None:
    store = SeriesStore(tmp_path / "series" / "saga_x", "saga_x")
    state, glossary = store.load_or_init()

    # Volume 1 saves Hero A portrait into persistent Portrait Vault
    man_v1 = _make_vol_manifest(tmp_path, "saga_x_v01", (40, 80, 160))
    seed_v1 = [
        SeedCharacter(
            canonical_name="Kuro",
            aliases=["Kuro-kun"],
            evidence_quote="Kuro, cẩn thận!",
            appearance_vi="tóc đen ngắn, áo khoác xanh",
            best_panel_id="ch01_p001_P1",
        )
    ]
    saved = store.save_character_portraits(state, seed_v1, man_v1, store.portraits_dir, current_vol=1)
    assert saved == 1
    kuro_rec = state.characters[0]
    assert Path(kuro_rec.portrait_path).exists()

    # Add 15 stale/deceased Vol 1 minor characters to simulate Volume 10 scale
    for i in range(15):
        state.characters.append(
            SeriesCharacter(
                char_id=f"c_old_{i:02d}",
                canonical_name=f"MinorChar{i}",
                status="deceased" if i < 8 else "active",
                first_seen_vol=1,
                last_seen_vol=1,
            )
        )
    state.last_handoff = VolumeHandoff(
        volume_index=9,
        job_id="saga_x_v09",
        ending_summary="Kuro đối đầu thủ lĩnh mới.",
        ending_characters=["Kuro"],
    )

    # Volume 10 has a completely different manifest (no Vol 1 panel_ids) + a new character
    man_v10 = _make_vol_manifest(tmp_path, "saga_x_v10", (120, 40, 60))
    man_v10.chapters[0].pages[0].panels[0].panel_id = "ch10_p001_P1"
    local_v10 = [
        SeedCharacter(
            canonical_name="Sora",
            evidence_quote="Tôi là Sora",
            appearance_vi="tóc bạc dài",
            best_panel_id="ch10_p001_P1",
        )
    ]
    active_cast = store.select_active_cast_for_volume(state, current_vol=10, local_seed_cast=local_v10, max_cards=16)
    names = [c.canonical_name for c in active_cast]
    assert "Sora" in names
    assert "Kuro" in names
    # Deceased and >4-vol-stale minor characters must NOT crowd out active cast
    assert all(not n.startswith("MinorChar") for n in names)

    # Kuro's entry carries Vault portrait_path & clears stale best_panel_id -> cast_sheet renders BOTH faces!
    kuro_seed = next(c for c in active_cast if c.canonical_name == "Kuro")
    assert kuro_seed.best_panel_id == ""
    assert kuro_seed.portrait_path == kuro_rec.portrait_path

    sheet = _build_cast_sheet(active_cast, man_v10, tmp_path / "cast_sheet_v10.jpg", max_cards=16)
    assert sheet is not None and sheet.exists()


def test_previously_revealed_character_not_blocked_by_spoiler_guard_in_later_volume() -> None:
    pages = [
        PageAnalysis(
            page_id="ch02_p001",
            chapter_id="ch02",
            page_summary="Kuro và kẻ lạ mặt chạm trán.",
            panels=[
                PanelAnalysis(
                    panel_id="ch02_p001_P1",
                    action_summary="Hai người đứng đối diện.",
                    dialogue_summary="Đừng lại gần!",
                    characters_present=["Kuro", "Vexor"],
                )
            ],
        ),
        PageAnalysis(
            page_id="ch02_p015",
            chapter_id="ch02",
            page_summary="Kẻ lạ xưng tên.",
            panels=[
                PanelAnalysis(
                    panel_id="ch02_p015_P1",
                    action_summary="Kẻ lạ bỏ mũ.",
                    dialogue_summary='Kẻ lạ nói: "Ta là Vexor, người gác cổng."',
                    characters_present=["Vexor"],
                )
            ],
        ),
    ]
    bible = StoryBible(
        characters=[
            CharacterProfile(canonical_name="Kuro"),
            CharacterProfile(canonical_name="Vexor"),
        ]
    )
    state = SeriesState(
        series_id="saga_x",
        characters=[
            SeriesCharacter(
                char_id="c001_kuro",
                canonical_name="Kuro",
                first_seen_vol=1,
                last_seen_vol=1,
                first_revealed_vol=1,
            )
        ],
    )
    already_revealed = SeriesStore.get_previously_revealed_names(state, current_vol=2)
    _compute_character_reveal_pages(pages, bible, already_revealed_names=already_revealed)

    by_name = {c.canonical_name: c for c in bible.characters}
    # Kuro was revealed in Vol 1 -> free to name from page 1 of Vol 2
    assert by_name["Kuro"].first_named_page_id == ""
    # Vexor is brand new in Vol 2 -> locked until ch02_p015
    assert by_name["Vexor"].first_named_page_id == "ch02_p015"


def test_three_tier_bounded_memory_and_human_glossary_override(tmp_path: Path) -> None:
    store = SeriesStore(tmp_path / "series" / "long_saga", "long_saga")
    state, glossary = store.load_or_init()

    # Simulate 23 completed volumes in Cold Ledger
    for v in range(1, 24):
        state.ledger.append(
            VolumeLedgerEntry(
                volume_index=v,
                job_id=f"long_saga_v{v:02d}",
                title_vi=f"Hồi {v}",
                summary_vi=(
                    "Nhân vật Ren trao thanh kiếm cổ ở tháp pha lê."
                    if v == 3
                    else f"Diễn biến bình thường của tập {v}."
                ),
                key_milestones=[f"Biến cố tập {v}"],
                characters_featured=["Ren"] if v == 3 else [f"Hero{v}"],
            )
        )
    state.characters.append(
        SeriesCharacter(char_id="c001_old", canonical_name="OldName", first_seen_vol=1, last_seen_vol=23)
    )
    state.active_plot_threads = ["Lời nguyền tháp pha lê vẫn chưa được hóa giải"]
    state.last_handoff = VolumeHandoff(
        volume_index=23,
        job_id="long_saga_v23",
        ending_summary="Cánh cổng tầng cuối mở ra trước mắt cả nhóm.",
        cliffhanger_hook="Ai đang chờ phía sau cánh cổng?",
        last_beats_tail=["Cánh cổng đá nặng nề hé mở trong bóng tối."],
        ending_characters=["Ren"],
    )
    store.save_state(state)

    # Human edits glossary.yaml between Vol 23 and Vol 24
    glossary.series_title = "Biên Niên Sử 24 Tập"
    glossary.terminology["Crystal Tower"] = "Tháp Pha Lê"
    glossary.pronoun_pairs["Ren -> Hana"] = "anh - em"
    glossary.character_name_overrides["OldName"] = "Ren"
    store.save_glossary(glossary)

    ctx = store.build_context(volume_index=24, total_volumes=24, continuity_mode="episodic")
    assert ctx.state.series_title == "Biên Niên Sử 24 Tập"
    assert ctx.state.characters[0].canonical_name == "Ren"
    assert "OldName" in ctx.state.characters[0].aliases

    mem_prompt = store.retrieve_bounded_memory_prompt(
        ctx,
        active_characters=["Ren"],
        current_text_hint="Ren cầm lại thanh kiếm cổ ở tháp pha lê",
        max_cold_callbacks=2,
    )
    # Must contain Hot Vol 23 + Warm Glossary + Targeted Cold callback to Vol 3, NOT all 23 volumes
    assert "Kết tập 23" in mem_prompt
    assert "Tháp Pha Lê" in mem_prompt
    assert "Tập 3" in mem_prompt
    assert "Tập 12" not in mem_prompt


def test_critic_series_continuity_gate_catches_glossary_drift_handoff_echo_and_seamless_greeting(
    tmp_path: Path,
) -> None:
    manifest = _make_vol_manifest(tmp_path, "saga_v02", (50, 50, 50))
    bible = StoryBible(series_title_guess="Saga", characters=[CharacterProfile(canonical_name="Ren")])
    state = SeriesState(
        series_id="saga",
        last_handoff=VolumeHandoff(
            volume_index=1,
            job_id="saga_v01",
            last_beats_tail=["Cánh cổng đá cổ xưa bỗng rung chuyển dữ dội trước mặt toàn đội."],
        ),
    )
    glossary = SeriesGlossary(
        terminology={"Crystal Tower": "Tháp Pha Lê"},
        character_name_overrides={"OldName": "Ren"},
    )
    ctx = SeriesContext(
        series_id="saga",
        series_dir=tmp_path / "series" / "saga",
        volume_index=2,
        total_volumes=4,
        continuity_mode="seamless",
        state=state,
        glossary=glossary,
    )
    script = RecapScript(
        title_vi="Tập 2",
        beats=[
            ScriptBeat(
                beat_id="b001",
                narration_vi="Chào mừng các bạn! Cánh cổng đá cổ xưa bỗng rung chuyển dữ dội trước mặt toàn đội.",
                panel_ids=["ch01_p001_P1"],
            ),
            ScriptBeat(
                beat_id="b002",
                narration_vi="OldName lao thẳng về phía Crystal Tower mà không do dự.",
                panel_ids=["ch01_p001_P1"],
            ),
        ],
    )
    critic = CriticAgent(LLMClient(LLMConfig(offline_fallback=True)), polish_seams=False)
    _, report = critic.review(script, manifest, bible, tmp_path / "job_v02", force=True, series_ctx=ctx)
    cont_msgs = [i.message for i in report.issues if i.category == "series_continuity"]
    assert any("seamless" in m for m in cont_msgs)
    assert any("lặp lại gần nguyên văn" in m for m in cont_msgs)
    assert any("Tháp Pha Lê" in m for m in cont_msgs)
    assert any("Ren" in m for m in cont_msgs)


def test_music_director_locks_series_palette_and_avoids_prev_vol_opener() -> None:
    catalog = MusicCatalog(
        tracks=[
            MusicTrack(
                track_id="tr_a",
                title="Track A",
                file_path="a.mp3",
                moods=["mysterious"],
                genres=["dark_fantasy"],
                energy=3,
                duration_sec=90.0,
            ),
            MusicTrack(
                track_id="tr_b",
                title="Track B",
                file_path="b.mp3",
                moods=["mysterious"],
                genres=["dark_fantasy"],
                energy=3,
                duration_sec=90.0,
            ),
        ]
    )
    cues = [MusicCue(cue_id="cue_01", start_ms=0, end_ms=30000, mood="mysterious")]
    picked_v1 = select_tracks_for_cues(cues, catalog, genre_prefs={"dark_fantasy": 1.0})
    first_id = picked_v1[0].track_id

    picked_v2 = select_tracks_for_cues(
        cues,
        catalog,
        genre_prefs={"dark_fantasy": 1.0},
        avoid_initial_track_ids=[first_id],
    )
    assert picked_v2[0].track_id != first_id


def test_series_compiler_merges_mp4_srt_chapters_and_credits(tmp_path: Path) -> None:
    assert parse_volume_range("1-3, 5", [1, 2, 3, 4, 5, 6]) == [1, 2, 3, 5]
    assert parse_volume_range(None, [3, 1, 2]) == [1, 2, 3]

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    jobs_dir = tmp_path / "jobs"
    state = SeriesState(series_id="demo_saga", series_title="Demo Saga")

    for v in (1, 2):
        jdir = jobs_dir / f"demo_saga_v{v:02d}"
        (jdir / "exports").mkdir(parents=True, exist_ok=True)
        mp4 = jdir / "06_output.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={'blue' if v == 1 else 'red'}:s=160x90:d=1.0:r=10",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-t",
                "1.0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(mp4),
            ],
            check=True,
        )
        (jdir / "06_subtitles.srt").write_text(
            f"1\n00:00:00,100 --> 00:00:00,900\nCâu tập {v}\n",
            encoding="utf-8",
        )
        (jdir / "exports" / "music_credits.txt").write_text(
            "# Music Credits\n\n- \"Shared Track\" by Artist — License: CC-BY\n"
            f"- \"Vol {v} Track\" by Artist — License: CC-BY\n",
            encoding="utf-8",
        )
        state.ledger.append(
            VolumeLedgerEntry(
                volume_index=v,
                job_id=f"demo_saga_v{v:02d}",
                title_vi=f"Biến cố tập {v}",
                duration_sec=1.0,
                output_mp4=str(mp4),
            )
        )

    res = compile_series_videos(
        state=state,
        jobs_dir=jobs_dir,
        out_dir=tmp_path / "series" / "demo_saga" / "compilations",
        volumes=[1, 2],
    )
    assert Path(res.output_mp4).exists() and Path(res.output_mp4).stat().st_size > 1000
    assert res.volumes_included == [1, 2]
    assert res.total_duration_sec >= 1.8

    srt_txt = Path(res.output_srt).read_text(encoding="utf-8")
    assert "Câu tập 1" in srt_txt and "Câu tập 2" in srt_txt
    assert "00:00:01," in srt_txt  # Vol 2 subtitle shifted by Vol 1 duration (~1000ms)

    chap_txt = Path(res.chapters_path).read_text(encoding="utf-8")
    assert "00:00 Tập 1: Biến cố tập 1" in chap_txt
    assert "Tập 2: Biến cố tập 2" in chap_txt

    cred_txt = Path(res.credits_path).read_text(encoding="utf-8")
    assert cred_txt.count("Shared Track") == 1
    assert "Vol 1 Track" in cred_txt and "Vol 2 Track" in cred_txt


def test_series_orchestrator_discovers_natural_order_and_runs_two_volumes(tmp_path: Path, monkeypatch) -> None:
    import wave
    from src.schemas.timeline import AudioBeatInfo, AudioManifest

    vols_root = tmp_path / "raw_manga"
    for name in ("vol10", "vol2", "vol1"):
        vdir = vols_root / name
        for p_i in range(1, 3):
            _make_panel_image(vdir / f"p{p_i:02d}.jpg", (30 * p_i, 60, 90))

    discovered = discover_series_volumes(vols_root)
    assert [p.name for _, p in discovered] == ["vol1", "vol2", "vol10"]

    def _fake_stage4(script, job_dir, tts_config, force=False):
        vo = job_dir / "04_voiceover.wav"
        vo.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(vo), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(b"\x00\x00" * 48000)
        beats = []
        cur = 0
        for b in script.beats:
            beats.append(
                AudioBeatInfo(
                    beat_id=b.beat_id,
                    audio_path=str(vo),
                    duration_ms=800,
                    start_ms=cur,
                    end_ms=cur + 800,
                    pause_after_ms=200,
                    sample_rate=48000,
                )
            )
            cur += 1000
        return AudioManifest(
            backend="mock_offline",
            voice="mock-vi",
            sample_rate=48000,
            total_duration_ms=max(1000, cur),
            voiceover_path=str(vo),
            beats=beats,
        )

    monkeypatch.setattr("src.pipeline.run_stage4_audio", _fake_stage4)

    cfg = AppConfig()
    cfg.project.jobs_dir = tmp_path / "jobs"
    cfg.project.series_dir = tmp_path / "series"
    cfg.llm.offline_fallback = True
    cfg.llm.api_key = None
    cfg.llm.base_url = None
    cfg.cv.detector = "manga_yolo"
    cfg.tts.backend = "edge_tts"

    # Run Vol 1..2 end-to-end offline (skip MP4 render for speed)
    orch = SeriesOrchestrator(cfg)
    rep = orch.run_series(
        input_dir=vols_root,
        series_id="test_saga",
        continuity_mode="seamless",
        from_vol=1,
        to_vol=2,
        skip_render=True,
    )
    assert rep.volumes_processed == [1, 2]

    store = SeriesStore(cfg.project.series_dir / "test_saga", "test_saga")
    state, glossary = store.load_or_init()
    assert [e.volume_index for e in state.ledger] == [1, 2]
    assert state.last_handoff is not None
    assert state.last_handoff.volume_index == 2
    assert store.glossary_path.exists()
    # Seamless mode on Vol 2 offline fallback uses continuation opening instead of YouTube greeting
    v2_script = RecapScript.model_validate_json(
        (cfg.project.jobs_dir / "test_saga_v02" / "03b_script_final.json").read_text(encoding="utf-8")
    )
    assert "Chào mừng" not in v2_script.beats[0].narration_vi
