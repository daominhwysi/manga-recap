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


_MOOD_PALETTES = {
    # Bright warm F-major / D-dorian pad + light bell pluck
    "warm": (87.31, (174.61, 220.00, 261.63), (349.23, 440.00, 523.25, 659.25), 0.55, 0.08),
    "calm": (73.42, (146.83, 220.00, 261.63), (293.66, 329.63, 440.00, 523.25), 0.65, 0.06),
    "comedic": (98.00, (196.00, 246.94, 293.66), (392.00, 493.88, 587.33, 440.00), 0.35, 0.11),
    # Dark fantasy gothic mystery / tension
    "mysterious": (73.42, (146.83, 174.61, 220.00), (293.66, 349.23, 440.00, 587.33), 0.50, 0.08),
    "tense": (65.41, (130.81, 155.56, 196.00), (261.63, 311.13, 392.00, 493.88), 0.38, 0.09),
    # Energetic battle pulse
    "action": (55.00, (110.00, 164.81, 220.00), (220.00, 261.63, 329.63, 440.00), 0.25, 0.13),
    # Slow tragic lament
    "tragic": (58.27, (116.54, 146.83, 220.00), (233.08, 293.66, 349.23, 440.00), 0.75, 0.06),
}


def build_dynamic_timeline_bgm(
    timeline,
    output_wav: Path,
    sample_rate: int = 48000,
) -> Path:
    """Synthesize continuous multi-mood ambient BGM + subtle SFX cues matching timeline shots."""
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    dur_sec = max(1.0, timeline.total_duration_ms / 1000.0)
    n_samples = int(round(dur_sec * sample_rate))
    t = np.linspace(0.0, dur_sec, n_samples, endpoint=False, dtype=np.float64)

    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    # Merge adjacent shots sharing the same mood into contiguous segments for smooth 1.2s crossfades
    segments: list[tuple[int, int, str]] = []
    for shot in timeline.shots:
        mood = shot.mood if shot.mood in _MOOD_PALETTES else "tense"
        if segments and segments[-1][2] == mood:
            s0, _, m0 = segments[-1]
            segments[-1] = (s0, shot.end_ms, m0)
        else:
            segments.append((shot.start_ms, shot.end_ms, mood))
    if not segments:
        segments = [(0, timeline.total_duration_ms, "tense")]

    xf_samples = int(1.0 * sample_rate)
    for idx, (s_ms, e_ms, mood) in enumerate(segments):
        i0 = max(0, int(round(s_ms / 1000.0 * sample_rate)) - (xf_samples // 2 if idx > 0 else 0))
        i1 = min(n_samples, int(round(e_ms / 1000.0 * sample_rate)) + (xf_samples // 2 if idx < len(segments) - 1 else 0))
        if i1 <= i0:
            continue
        seg_t = t[i0:i1]
        bass_hz, pad_freqs, arp_freqs, step_dur, chime_amp = _MOOD_PALETTES[mood]

        pad_l = 0.26 * np.sin(2.0 * np.pi * bass_hz * seg_t)
        pad_r = 0.26 * np.sin(2.0 * np.pi * bass_hz * seg_t + 0.15)
        for p_idx, f_hz in enumerate(pad_freqs):
            det = 1.0 + (0.0018 if p_idx % 2 == 0 else -0.0018)
            pad_l += 0.11 * np.sin(2.0 * np.pi * f_hz * seg_t)
            pad_r += 0.11 * np.sin(2.0 * np.pi * (f_hz * det) * seg_t + 0.35)

        step_idx = np.floor(seg_t / step_dur).astype(np.int64)
        step_t = seg_t - step_idx * step_dur
        note_f = np.array([arp_freqs[s % len(arp_freqs)] for s in step_idx], dtype=np.float64)
        plucked = np.exp(-step_t * 6.0) * np.minimum(1.0, step_t / 0.008)
        chime = chime_amp * plucked * np.sin(2.0 * np.pi * note_f * seg_t)

        seg_l = pad_l + chime * 0.85
        seg_r = pad_r + chime * 0.65

        # Raised-cosine fade in/out weights at segment joins
        w = np.ones(i1 - i0, dtype=np.float64)
        if idx > 0:
            fin = min(xf_samples, len(w) // 2)
            if fin > 0:
                w[:fin] *= np.linspace(0.0, 1.0, fin)
        if idx < len(segments) - 1:
            fout = min(xf_samples, len(w) // 2)
            if fout > 0:
                w[-fout:] *= np.linspace(1.0, 0.0, fout)
        left[i0:i1] += seg_l * w
        right[i0:i1] += seg_r * w

    # Subtle SFX accents synchronized with shake_impact & flash_white shots
    for shot in timeline.shots:
        s_idx = int(round(shot.start_ms / 1000.0 * sample_rate))
        if shot.camera_move == "shake_impact" and 0 <= s_idx < n_samples:
            slen = min(int(0.28 * sample_rate), n_samples - s_idx)
            if slen > 64:
                st = np.linspace(0.0, slen / sample_rate, slen, endpoint=False)
                thud = 0.32 * np.exp(-st * 14.0) * np.sin(2.0 * np.pi * (95.0 - 50.0 * st) * st)
                left[s_idx : s_idx + slen] += thud
                right[s_idx : s_idx + slen] += thud
        elif shot.transition_in == "flash_white" and 0 <= s_idx < n_samples:
            slen = min(int(0.22 * sample_rate), n_samples - s_idx)
            if slen > 64:
                st = np.linspace(0.0, slen / sample_rate, slen, endpoint=False)
                sting = 0.16 * np.exp(-st * 12.0) * np.sin(2.0 * np.pi * 880.0 * st)
                left[s_idx : s_idx + slen] += sting * 0.8
                right[s_idx : s_idx + slen] += sting * 0.8

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) or 1.0
    stereo = (stereo / peak) * 0.55
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return output_wav
