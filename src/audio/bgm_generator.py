"""Synthesizes a seamless dark-fantasy ambient stereo BGM loop tailored for voiceover mixing."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def ensure_default_bgm(
    bgm_path: Path,
    duration_sec: float = 64.0,
    sample_rate: int = 48000,
) -> Path:
    """Create a copyright-free D-minor gothic clockwork + warm pad loop if missing."""
    bgm_path = Path(bgm_path)
    if bgm_path.exists() and bgm_path.stat().st_size > 4096:
        return bgm_path

    bgm_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float64)

    # 4-chord progression across 16s bar cycle (loops seamlessly over 64s = 4 cycles):
    # Dm (D3-F3-A3), Bb (Bb2-D3-F3), Gm (G2-Bb2-D3), A (A2-C#3-E3)
    chords = [
        (73.42, (146.83, 174.61, 220.00), (293.66, 349.23, 440.00, 587.33)),
        (58.27, (116.54, 146.83, 174.61), (233.08, 293.66, 349.23, 466.16)),
        (98.00, (130.81, 155.56, 196.00), (196.00, 233.08, 293.66, 392.00)),
        (55.00, (110.00, 138.59, 164.81), (220.00, 277.18, 329.63, 440.00)),
    ]

    cycle_sec = 16.0
    seg_sec = 4.0
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    for c_idx, (bass_hz, pad_freqs, arp_freqs) in enumerate(chords):
        phase_in_cycle = np.mod(t, cycle_sec)
        local_t = phase_in_cycle - c_idx * seg_sec
        mask = (local_t >= 0.0) & (local_t < seg_sec)
        if not np.any(mask):
            continue

        lt = local_t[mask]
        # Smooth raised-cosine pad envelope per chord bar so chord changes never click
        pad_env = np.sin(np.pi * lt / seg_sec) ** 0.45
        fade_tail = 0.35 + 0.65 * pad_env

        # Warm sub bass + octave shadow
        bass = 0.28 * np.sin(2.0 * np.pi * bass_hz * t[mask]) + 0.10 * np.sin(
            2.0 * np.pi * (bass_hz * 2.0) * t[mask]
        )

        # Warm dark minor pad kept below vocal presence range (110-440 Hz)
        pad_l = np.zeros_like(lt)
        pad_r = np.zeros_like(lt)
        for p_idx, f_hz in enumerate(pad_freqs):
            detune = 1.0 + (0.0018 if p_idx % 2 == 0 else -0.0018)
            pad_l += 0.11 * np.sin(2.0 * np.pi * f_hz * t[mask])
            pad_r += 0.11 * np.sin(2.0 * np.pi * (f_hz * detune) * t[mask] + 0.4)

        # Gentle 0.5s eighth-note music-box clockwork arpeggio (low amplitude so voice stays dominant)
        step_dur = 0.5
        step_idx = np.floor(lt / step_dur).astype(np.int64)
        step_t = lt - step_idx * step_dur
        note_f = np.array([arp_freqs[s % len(arp_freqs)] for s in step_idx], dtype=np.float64)
        plucked_env = np.exp(-step_t * 6.5) * np.minimum(1.0, step_t / 0.008)
        chime = 0.075 * plucked_env * (
            np.sin(2.0 * np.pi * note_f * t[mask])
            + 0.22 * np.sin(2.0 * np.pi * (note_f * 2.0) * t[mask])
        )
        pan_l = np.where((step_idx % 2) == 0, 0.85, 0.55)
        pan_r = np.where((step_idx % 2) == 0, 0.55, 0.85)

        left[mask] = (bass + pad_l) * fade_tail + chime * pan_l
        right[mask] = (bass + pad_r) * fade_tail + chime * pan_r

    # Subtle 1.0s seamless loop crossfade at start/end boundaries
    xf_len = min(n_samples // 10, int(1.0 * sample_rate))
    if xf_len > 0:
        fade_in = np.linspace(0.0, 1.0, xf_len)
        fade_out = 1.0 - fade_in
        left[:xf_len] = left[:xf_len] * fade_in + left[-xf_len:] * fade_out
        right[:xf_len] = right[:xf_len] * fade_in + right[-xf_len:] * fade_out

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) or 1.0
    stereo = (stereo / peak) * 0.55
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(bgm_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return bgm_path
