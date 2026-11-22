"""Shared audio helpers and base class for TTS backends."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from src.schemas.timeline import AudioBeatInfo, WordTimestamp


def apply_pronunciation_glossary(text: str, pronunciation_map: dict[str, str] | None) -> str:
    """Normalize foreign terms/names using pronunciation_map for TTS while preserving display subtitles."""
    if not text or not pronunciation_map:
        return text
    out = text
    for term, spoken in sorted(pronunciation_map.items(), key=lambda kv: -len(kv[0])):
        clean_t = term.strip()
        clean_s = spoken.strip()
        if not clean_t or not clean_s or clean_t.lower() == clean_s.lower():
            continue
        pat = re.compile(rf"(?<!\w){re.escape(clean_t)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
        out = pat.sub(clean_s, out)
    return out


def get_wav_duration_ms(wav_path: Path) -> int:
    info = sf.info(str(wav_path))
    return int(round((info.frames / max(1, info.samplerate)) * 1000))


def beat_cache_key(
    beat_id: str,
    text: str,
    voice: str,
    backend: str,
    rate: str = "+0%",
) -> str:
    rate_part = f"|{rate}" if (backend == "edge_tts" or rate != "+0%") else ""
    digest = hashlib.sha256(f"{backend}|{voice}{rate_part}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{beat_id}_{digest}.wav"


def resample_audio_antialias(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Bandlimited Fourier resampling with Nyquist low-pass roll-off to avoid aliasing distortion."""
    n = len(data)
    if orig_sr == target_sr or n <= 1:
        return data.astype(np.float32)
    new_len = max(1, int(round(n * target_sr / orig_sr)))
    spec = np.fft.rfft(data.astype(np.float64))
    n_bins = len(spec)
    # Apply gentle raised-cosine anti-aliasing taper near the lower Nyquist limit
    cutoff_bin = max(4, min(n_bins - 1, int(n_bins * min(1.0, target_sr / max(1, orig_sr)) * 0.92)))
    taper_start = max(1, int(cutoff_bin * 0.88))
    if cutoff_bin > taper_start:
        taper = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, cutoff_bin - taper_start)))
        spec[taper_start:cutoff_bin] *= taper
    spec[cutoff_bin:] = 0.0

    out_bins = new_len // 2 + 1
    out_spec = np.zeros(out_bins, dtype=np.complex128)
    copy_bins = min(len(spec), out_bins)
    out_spec[:copy_bins] = spec[:copy_bins]
    resampled = np.fft.irfft(out_spec, n=new_len) * (new_len / float(n))
    return np.clip(resampled, -1.0, 1.0).astype(np.float32)


def estimate_word_timestamps_from_wav(
    wav_path: Path,
    text: str,
    start_offset_ms: int = 0,
) -> list[WordTimestamp]:
    """Estimate per-word timestamps using acoustic phrase/pause boundaries when engine omits them."""
    words = text.strip().split()
    if not words:
        return []
    try:
        data, sr = sf.read(str(wav_path), dtype="float32")
    except Exception:
        return []
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    total_ms = max(200, int(round(len(data) / max(1, sr) * 1000)))

    # Group words into punctuation clauses and weight durations with slight pause allowances
    weights: list[float] = []
    for w in words:
        base = max(2.0, float(len(re.sub(r"[^\w]", "", w, flags=re.UNICODE))))
        if re.search(r"[.!?…]$", w):
            base += 4.5
        elif re.search(r"[,;:]$", w):
            base += 2.2
        weights.append(base)
    w_sum = sum(weights) or float(len(words))

    timestamps: list[WordTimestamp] = []
    cur_ms = 0
    for idx, (word, wt) in enumerate(zip(words, weights)):
        span = (
            total_ms - cur_ms
            if idx == len(words) - 1
            else max(35, int(round(total_ms * wt / w_sum)))
        )
        w_end = min(total_ms, cur_ms + span)
        timestamps.append(
            WordTimestamp(
                word=word,
                start_ms=start_offset_ms + cur_ms,
                end_ms=start_offset_ms + max(cur_ms + 30, w_end),
            )
        )
        cur_ms = w_end
    return timestamps


def concat_beat_wavs(
    beat_infos: list[AudioBeatInfo],
    output_wav: Path,
    target_sr: int = 48000,
) -> tuple[int, list[AudioBeatInfo]]:
    """Concatenate per-beat WAV files with inter-beat silence pauses and aligned word timestamps."""
    chunks: list[np.ndarray] = []
    cursor_ms = 0
    updated: list[AudioBeatInfo] = []

    for info in beat_infos:
        data, sr = sf.read(info.audio_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr and len(data) > 0:
            data = resample_audio_antialias(data, orig_sr=sr, target_sr=target_sr)

        dur_ms = int(round(len(data) / target_sr * 1000))
        start_ms = cursor_ms
        end_ms = start_ms + dur_ms
        chunks.append(data)

        pause_samples = int(round(info.pause_after_ms * target_sr / 1000))
        if pause_samples > 0:
            chunks.append(np.zeros(pause_samples, dtype=np.float32))
        cursor_ms = end_ms + info.pause_after_ms

        shifted_wts = [
            WordTimestamp(
                word=wt.word,
                start_ms=start_ms + wt.start_ms,
                end_ms=start_ms + wt.end_ms,
            )
            for wt in info.word_timestamps
        ]

        updated.append(
            AudioBeatInfo(
                beat_id=info.beat_id,
                audio_path=info.audio_path,
                duration_ms=dur_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                pause_after_ms=info.pause_after_ms,
                sample_rate=target_sr,
                word_timestamps=shifted_wts,
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
