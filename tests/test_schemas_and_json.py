"""Tests for Pydantic schemas and JSON round-trip serialization."""

from __future__ import annotations

from src.schemas import (
    AudioBeatInfo,
    AudioManifest,
    BBox,
    JobManifest,
    RecapScript,
    ScriptBeat,
    StoryBible,
    TimelineShot,
    VideoTimeline,
)


def test_bbox_geometry() -> None:
    b = BBox(x1=10, y1=20, x2=110, y2=220)
    assert b.width == 100
    assert b.height == 200
    assert b.area == 20000


def test_schema_roundtrip() -> None:
    script = RecapScript(
        title_vi="Test",
        beats=[ScriptBeat(beat_id="b001", narration_vi="Xin chào", panel_ids=["p1"])],
    )
    loaded = RecapScript.model_validate_json(script.model_dump_json())
    assert loaded.beats[0].beat_id == "b001"

    tl = VideoTimeline(
        job_id="j1",
        width=1280,
        height=720,
        fps=24,
        total_duration_ms=2000,
        voiceover_path="vo.wav",
        shots=[
            TimelineShot(
                shot_id="s1",
                beat_id="b001",
                panel_id="p1",
                panel_path="p1.jpg",
                start_ms=0,
                end_ms=2000,
            )
        ],
    )
    tl2 = VideoTimeline.model_validate_json(tl.model_dump_json())
    assert tl2.shots[0].end_ms == 2000
    assert JobManifest(job_id="x", input_path="in").job_id == "x"
    assert StoryBible().series_title_guess == "Manga Recap"
    assert AudioManifest(
        backend="edge_tts",
        voice="v",
        total_duration_ms=100,
        voiceover_path="a.wav",
        beats=[AudioBeatInfo(beat_id="b1", audio_path="b1.wav", duration_ms=100)],
    ).total_duration_ms == 100
