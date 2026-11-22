"""Tests for audio stitching and silence/duration utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.base import (
    BaseTTSBackend,
    apply_pronunciation_glossary,
    beat_cache_key,
    concat_beat_wavs,
    get_wav_duration_ms,
    resample_audio_antialias,
)
from src.audio.bgm_generator import build_dynamic_timeline_bgm
from src.schemas.timeline import AudioBeatInfo, TimelineShot, VideoTimeline


def test_concat_beat_wavs(tmp_path: Path) -> None:
    sr = 48000
    b1 = tmp_path / "b1.wav"
    b2 = tmp_path / "b2.wav"
    sf.write(str(b1), np.zeros(sr // 2, dtype=np.float32), sr, subtype="PCM_16")
    sf.write(str(b2), np.ones(sr // 2, dtype=np.float32) * 0.1, sr, subtype="PCM_16")

    assert get_wav_duration_ms(b1) == 500
    out_wav = tmp_path / "merged.wav"
    total_ms, timed = concat_beat_wavs(
        [
            AudioBeatInfo(beat_id="b1", audio_path=str(b1), duration_ms=500, pause_after_ms=200),
            AudioBeatInfo(beat_id="b2", audio_path=str(b2), duration_ms=500, pause_after_ms=100),
        ],
        out_wav,
        target_sr=sr,
    )
    assert out_wav.exists()
    assert total_ms == 1300
    assert timed[1].start_ms == 700
    assert BaseTTSBackend.write_silence_wav(tmp_path / "sil.wav", 300, sr) == 300


def test_pronunciation_cache_rate_antialias_and_dynamic_bgm(tmp_path: Path) -> None:
    # 1) Pronunciation normalizer
    assert apply_pronunciation_glossary("Arthur rút kiếm trong Abyss.", {"Arthur": "A thơ", "Abyss": "A bít"}) == "A thơ rút kiếm trong A bít."

    # 2) beat_cache_key separates different Edge-TTS rates
    k1 = beat_cache_key("b1", "Xin chào", "vi-VN-NamMinhNeural", "edge_tts", rate="+0%")
    k2 = beat_cache_key("b1", "Xin chào", "vi-VN-NamMinhNeural", "edge_tts", rate="+15%")
    assert k1 != k2

    # 3) Anti-aliased resampling produces finite target length
    arr = np.sin(np.linspace(0, 2 * np.pi * 10, 2400)).astype(np.float32)
    up = resample_audio_antialias(arr, 24000, 48000)
    assert len(up) == 4800
    assert np.isfinite(up).all()

    # 4) Multi-mood dynamic BGM synthesis
    tl = VideoTimeline(
        job_id="bgm_test",
        width=1280,
        height=720,
        fps=24,
        total_duration_ms=2000,
        voiceover_path="vo.wav",
        shots=[
            TimelineShot(
                shot_id="s1", beat_id="b1", panel_id="p1", panel_path="p1.jpg",
                start_ms=0, end_ms=1000, mood="calm",
            ),
            TimelineShot(
                shot_id="s2", beat_id="b2", panel_id="p2", panel_path="p2.jpg",
                start_ms=1000, end_ms=2000, camera_move="shake_impact", transition_in="flash_white", mood="action",
            ),
        ],
    )
    bgm_out = build_dynamic_timeline_bgm(tl, tmp_path / "dyn_bgm.wav")
    assert bgm_out.exists() and bgm_out.stat().st_size > 10000
