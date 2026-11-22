"""Tests for config loading (including default_model=gpt-6-luna) and registries."""

from __future__ import annotations

from pathlib import Path

from src.audio.registry import TTS_REGISTRY
from src.config import load_config, load_personas
from src.cv.registry import DETECTOR_REGISTRY
from src.exporters.registry import EXPORTER_REGISTRY
from src.render.registry import RENDERER_REGISTRY


def test_load_default_config() -> None:
    cfg = load_config(Path("config.yaml"))
    assert cfg.llm.default_model == "gpt-6-luna"
    assert cfg.llm.provider == "aixingialaire"
    assert cfg.llm.vision_provider is None
    assert cfg.script.polish_seams is False
    assert cfg.render.bgm_path == "assets/bgm/default_dark_fantasy.wav"
    assert cfg.render.audio_channels == 2
    assert cfg.project.jobs_dir == Path("jobs")


def test_default_bgm_generator(tmp_path: Path) -> None:
    import wave
    from src.audio.bgm_generator import ensure_default_bgm

    out = ensure_default_bgm(tmp_path / "bgm" / "default_dark_fantasy.wav", duration_sec=4.0)
    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getframerate() == 48000
        assert wf.getnframes() == 48000 * 4


def test_personas_loaded() -> None:
    personas = load_personas(Path("configs/personas.yaml"))
    assert "kich_tinh" in personas
    assert "hai_huoc" in personas
    assert "phan_tich" in personas


def test_registries_populated() -> None:
    import src.audio  # noqa: F401
    import src.cv  # noqa: F401
    import src.exporters  # noqa: F401
    import src.render  # noqa: F401

    assert "manga_yolo" in DETECTOR_REGISTRY.list_names()
    assert "manhwa_slicer" in DETECTOR_REGISTRY.list_names()
    assert "edge_tts" in TTS_REGISTRY.list_names()
    assert "vieneu" in TTS_REGISTRY.list_names()
    assert "ffmpeg_opencv" in RENDERER_REGISTRY.list_names()
    assert "capcut_draft" in EXPORTER_REGISTRY.list_names()
    assert "srt" in EXPORTER_REGISTRY.list_names()
