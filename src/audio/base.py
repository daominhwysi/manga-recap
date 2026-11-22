"""Shared audio helpers and base class for TTS backends."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from src.schemas.timeline import AudioBeatInfo


def get_wav_duration_ms(wav_path: Path) -> int:
    info = sf.info(str(wav_path))
    return int(round((info.frames / max(1, info.samplerate)) * 1000))


def beat_cache_key(beat_id: str, text: str, voice: str, backend: str) -> str:
    digest = hashlib.sha256(f"{backend}|{voice}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{beat_id}_{digest}.wav"


def concat_beat_wavs(
    beat_infos: list[AudioBeatInfo],
    output_wav: Path,
    target_sr: int = 48000,
) -> tuple[int, list[AudioBeatInfo]]:
    """Concatenate per-beat WAV files with inter-beat silence pauses."""
    chunks: list[np.ndarray] = []
    cursor_ms = 0
    updated: list[AudioBeatInfo] = []

    for info in beat_infos:
        data, sr = sf.read(info.audio_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr and len(data) > 0:
            new_len = int(round(len(data) * target_sr / sr))
            x_old = np.linspace(0.0, 1.0, len(data), endpoint=False)
            x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)

        dur_ms = int(round(len(data) / target_sr * 1000))
        start_ms = cursor_ms
        end_ms = start_ms + dur_ms
        chunks.append(data)

        pause_samples = int(round(info.pause_after_ms * target_sr / 1000))
        if pause_samples > 0:
            chunks.append(np.zeros(pause_samples, dtype=np.float32))
        cursor_ms = end_ms + info.pause_after_ms

        updated.append(
            AudioBeatInfo(
                beat_id=info.beat_id,
                audio_path=info.audio_path,
                duration_ms=dur_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                pause_after_ms=info.pause_after_ms,
                sample_rate=target_sr,
            )
        )

    merged = np.concatenate(chunks) if chunks else np.zeros(target_sr, dtype=np.float32)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), merged, target_sr, subtype="PCM_16")
    return cursor_ms, updated


class BaseTTSBackend:
    """Base helper for TTS backends."""

    name: str = "base"

    @staticmethod
    def write_silence_wav(output_path: Path, duration_ms: int = 1500, sample_rate: int = 48000) -> int:
        samples = np.zeros(int(sample_rate * duration_ms / 1000), dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), samples, sample_rate, subtype="PCM_16")
        return duration_ms
