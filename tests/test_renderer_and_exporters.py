"""Tests for FFmpeg+OpenCV video rendering (with stereo audio) and exporters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf

from src.exporters.capcut_draft import CapCutDraftExporter
from src.exporters.html_preview import HtmlPreviewExporter
from src.exporters.srt_exporter import SrtExporter
from src.render.ffmpeg_renderer import FFmpegOpenCVRenderer
from src.schemas.timeline import SubtitleCue, TimelineShot, VideoTimeline


def test_render_mp4_with_stereo_audio_and_exporters(tmp_path: Path) -> None:
    panel = tmp_path / "panel.jpg"
    cv2.imwrite(str(panel), np.full((300, 300, 3), 120, dtype=np.uint8))

    vo = tmp_path / "vo.wav"
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(vo), tone, sr, subtype="PCM_16")

    timeline = VideoTimeline(
        job_id="test_render",
        width=320,
        height=180,
        fps=12,
        total_duration_ms=1000,
        voiceover_path=str(vo),
        shots=[
            TimelineShot(
                shot_id="b001_s1",
                beat_id="b001",
                panel_id="p1",
                panel_path=str(panel),
                start_ms=0,
                end_ms=1000,
                camera_move="zoom_in",
            )
        ],
        subtitles=[SubtitleCue(index=1, start_ms=0, end_ms=900, text="Xin chào thế giới")],
    )

    out_mp4 = tmp_path / "out.mp4"
    renderer = FFmpegOpenCVRenderer(audio_channels=2)
    renderer.render(timeline, out_mp4)
    assert out_mp4.exists() and out_mp4.stat().st_size > 1000

    # Verify stereo audio stream exists in rendered MP4
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ff, "-i", str(out_mp4)], capture_output=True, text=True).stderr
    assert "Audio: aac" in probe
    assert "stereo" in probe

    # Exporters
    srt = SrtExporter().export(timeline, tmp_path)
    assert srt.exists() and "Xin chào thế giới" in srt.read_text(encoding="utf-8")
    draft = CapCutDraftExporter().export(timeline, tmp_path)
    assert (draft / "draft_content.json").exists()
    data = json.loads((draft / "draft_content.json").read_text(encoding="utf-8"))
    assert len(data["tracks"]) == 2
    html = HtmlPreviewExporter().export(timeline, tmp_path)
    assert html.exists()


def test_diptych_transitions_and_ducked_bgm_render(tmp_path: Path) -> None:
    p1 = tmp_path / "p1.jpg"
    p2 = tmp_path / "p2.jpg"
    cv2.imwrite(str(p1), np.full((320, 160, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(p2), np.full((320, 160, 3), 180, dtype=np.uint8))

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    vo = tmp_path / "vo2.wav"
    bgm = tmp_path / "bgm.wav"
    sf.write(str(vo), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, subtype="PCM_16")
    sf.write(str(bgm), (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")

    timeline = VideoTimeline(
        job_id="test_diptych",
        width=320,
        height=180,
        fps=12,
        total_duration_ms=1000,
        voiceover_path=str(vo),
        bgm_path=str(bgm),
        shots=[
            TimelineShot(
                shot_id="b001_s1",
                beat_id="b001",
                panel_id="p1",
                panel_path=str(p1),
                secondary_panel_id="p2",
                secondary_panel_path=str(p2),
                start_ms=0,
                end_ms=500,
                camera_move="diptych",
                transition_in="cut",
            ),
            TimelineShot(
                shot_id="b002_s1",
                beat_id="b002",
                panel_id="p2",
                panel_path=str(p2),
                start_ms=500,
                end_ms=1000,
                camera_move="pan_down",
                transition_in="crossfade",
            ),
        ],
        subtitles=[],
    )

    out_mp4 = tmp_path / "out_bgm.mp4"
    renderer = FFmpegOpenCVRenderer(audio_channels=2, bgm_volume=0.14, bgm_ducking=True)
    renderer.render(timeline, out_mp4)
    assert out_mp4.exists() and out_mp4.stat().st_size > 1000


def test_choreographer_skips_slivers_and_duplicate_diptych() -> None:
    from src.agents.choreographer import _is_diptych_candidate, _is_good_visual_panel
    from src.schemas.ingestion import BBox, FocalPoint, PanelInfo

    sliver = PanelInfo(
        panel_id="p48_P7",
        chapter_id="ch01",
        page_id="ch01_p048",
        index_in_page=7,
        global_index=7,
        bbox=BBox(x1=0, y1=0, x2=78, y2=700),
        width=78,
        height=700,
        crop_path="x.jpg",
        focal_points=[],
    )
    good = PanelInfo(
        panel_id="p48_P3",
        chapter_id="ch01",
        page_id="ch01_p048",
        index_in_page=3,
        global_index=3,
        bbox=BBox(x1=0, y1=0, x2=320, y2=680),
        width=320,
        height=680,
        crop_path="y.jpg",
        focal_points=[
            FocalPoint(
                kind="face",
                bbox=BBox(x1=40, y1=40, x2=180, y2=180),
                norm_cx=0.35,
                norm_cy=0.16,
                norm_w=0.4,
                norm_h=0.2,
                confidence=0.9,
            )
        ],
    )
    assert not _is_good_visual_panel(sliver)
    assert _is_good_visual_panel(good)
    assert not _is_diptych_candidate(good, good)


def test_trim_excess_white_border() -> None:
    from src.render.ffmpeg_renderer import _trim_excess_white_border

    canvas = np.full((800, 600, 3), 255, dtype=np.uint8)
    # Inset dark illustration box occupying ~12% of page area
    cv2.rectangle(canvas, (300, 400), (540, 640), (20, 20, 20), -1)
    trimmed = _trim_excess_white_border(canvas)
    assert trimmed.shape[0] < 450
    assert trimmed.shape[1] < 450


def test_choreographer_same_page_diptych_and_no_cross_page_borrow(tmp_path: Path) -> None:
    from src.agents.choreographer import ChoreographerAgent, _is_diptych_candidate
    from src.config import RenderConfig
    from src.schemas.ingestion import BBox, ChapterInfo, JobManifest, PageInfo, PanelInfo
    from src.schemas.script import RecapScript, ScriptBeat
    from src.schemas.timeline import AudioBeatInfo, AudioManifest

    def _pnl(pid: str, page_id: str, gidx: int, w: int = 180, h: int = 400) -> PanelInfo:
        return PanelInfo(
            panel_id=pid,
            chapter_id="ch01",
            page_id=page_id,
            index_in_page=gidx,
            global_index=gidx,
            bbox=BBox(x1=0, y1=0, x2=w, y2=h),
            width=w,
            height=h,
            crop_path=f"{pid}.jpg",
        )

    p1 = _pnl("ch01_p001_P1", "ch01_p001", 1)
    p2_other_page = _pnl("ch01_p002_P1", "ch01_p002", 2)
    assert not _is_diptych_candidate(p1, p2_other_page)

    # Wide beat duration needing extra slots must NOT borrow from ch01_p002 when beat only references ch01_p001
    p1_b = _pnl("ch01_p001_P2", "ch01_p001", 2, w=320, h=320)
    p2_b = _pnl("ch01_p002_P1", "ch01_p002", 3, w=320, h=320)
    manifest = JobManifest(
        job_id="j_choreo",
        input_path="in",
        chapters=[
            ChapterInfo(
                chapter_id="ch01",
                title="ch01",
                pages=[
                    PageInfo(page_id="ch01_p001", chapter_id="ch01", page_index=1, source_path="p1.jpg", width=400, height=600, panels=[p1, p1_b]),
                    PageInfo(page_id="ch01_p002", chapter_id="ch01", page_index=2, source_path="p2.jpg", width=400, height=600, panels=[p2_b]),
                ],
            )
        ],
    )
    script = RecapScript(
        title_vi="T",
        beats=[
            ScriptBeat(beat_id="b001", narration_vi="Câu một ngắn. Câu hai dài hơn một chút để thử chia nhịp.", panel_ids=["ch01_p001_P1"]),
        ],
    )
    audio = AudioManifest(
        backend="edge_tts",
        voice="v",
        total_duration_ms=9000,
        voiceover_path="vo.wav",
        beats=[AudioBeatInfo(beat_id="b001", audio_path="b1.wav", start_ms=0, end_ms=8800, duration_ms=8800, pause_after_ms=200)],
    )
    tl = ChoreographerAgent(RenderConfig()).choreograph("j_choreo", manifest, script, audio, tmp_path)
    used_panels = {s.panel_id for s in tl.shots} | {s.secondary_panel_id for s in tl.shots if s.secondary_panel_id}
    assert "ch01_p002_P1" not in used_panels
    assert "ch01_p001_P2" in used_panels
