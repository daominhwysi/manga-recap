"""Tests for audio stitching and silence/duration utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.base import BaseTTSBackend, concat_beat_wavs, get_wav_duration_ms
from src.schemas.timeline import AudioBeatInfo


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
