"""Synthesizes rich multi-mood BGM progressions, layered SFX, and optional real mood-track loading."""

from __future__ import annotations

import subprocess
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

    # 4-chord progression across 16s bar cycle: Dm - Bb - Gm - A7
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
        pad_env = np.sin(np.pi * lt / seg_sec) ** 0.45
        fade_tail = 0.35 + 0.65 * pad_env

        bass = 0.28 * np.sin(2.0 * np.pi * bass_hz * t[mask]) + 0.10 * np.sin(
            2.0 * np.pi * (bass_hz * 2.0) * t[mask]
        )
        pad_l = np.zeros_like(lt)
        pad_r = np.zeros_like(lt)
        for p_idx, f_hz in enumerate(pad_freqs):
            detune = 1.0 + (0.0018 if p_idx % 2 == 0 else -0.0018)
            pad_l += 0.11 * np.sin(2.0 * np.pi * f_hz * t[mask])
            pad_r += 0.11 * np.sin(2.0 * np.pi * (f_hz * detune) * t[mask] + 0.4)

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


# 4-bar chord progressions per mood: (bass_hz, pad_triad, motif_notes)
_MOOD_PROGRESSIONS: dict[str, list[tuple[float, tuple[float, float, float], tuple[float, ...]]]] = {
    "warm": [  # F -> C -> Dm -> Bb (gentle acoustic/Rhodes feel)
        (87.31, (174.61, 220.00, 261.63), (349.23, 440.00, 523.25, 440.00)),
        (65.41, (164.81, 196.00, 261.63), (329.63, 392.00, 523.25, 392.00)),
        (73.42, (146.83, 174.61, 220.00), (293.66, 349.23, 440.00, 349.23)),
        (58.27, (146.83, 174.61, 233.08), (293.66, 349.23, 466.16, 349.23)),
    ],
    "calm": [  # Dm9 -> Bbmaj7 -> Gm7 -> A7sus4
        (73.42, (146.83, 174.61, 220.00), (329.63, 440.00, 523.25, 440.00)),
        (58.27, (146.83, 174.61, 220.00), (293.66, 349.23, 440.00, 349.23)),
        (98.00, (146.83, 174.61, 196.00), (293.66, 349.23, 392.00, 349.23)),
        (55.00, (146.83, 164.81, 220.00), (293.66, 329.63, 440.00, 329.63)),
    ],
    "comedic": [  # G -> D -> Em -> C playful bouncy bounce
        (98.00, (196.00, 246.94, 293.66), (392.00, 493.88, 587.33, 493.88)),
        (73.42, (146.83, 185.00, 220.00), (369.99, 440.00, 587.33, 440.00)),
        (82.41, (164.81, 196.00, 246.94), (329.63, 392.00, 493.88, 392.00)),
        (65.41, (130.81, 164.81, 196.00), (329.63, 392.00, 523.25, 392.00)),
    ],
    "mysterious": [  # Dm -> Gm -> Bb -> A7 gothic curiosity
        (73.42, (146.83, 174.61, 220.00), (293.66, 349.23, 440.00, 587.33)),
        (98.00, (146.83, 196.00, 233.08), (293.66, 392.00, 466.16, 392.00)),
        (58.27, (146.83, 174.61, 233.08), (293.66, 349.23, 466.16, 349.23)),
        (55.00, (138.59, 164.81, 220.00), (277.18, 329.63, 440.00, 329.63)),
    ],
    "tense": [  # Cm -> Ab -> Fm -> G7 suspense pulse
        (65.41, (130.81, 155.56, 196.00), (261.63, 311.13, 392.00, 311.13)),
        (51.91, (130.81, 155.56, 207.65), (261.63, 311.13, 415.30, 311.13)),
        (87.31, (130.81, 174.61, 207.65), (261.63, 349.23, 415.30, 349.23)),
        (98.00, (123.47, 146.83, 196.00), (246.94, 293.66, 392.00, 493.88)),
    ],
    "action": [  # Am -> F -> G -> Em energetic driving ostinato
        (55.00, (110.00, 164.81, 220.00), (220.00, 329.63, 440.00, 523.25)),
        (87.31, (130.81, 174.61, 220.00), (261.63, 349.23, 440.00, 523.25)),
        (98.00, (146.83, 196.00, 246.94), (293.66, 392.00, 493.88, 587.33)),
        (82.41, (123.47, 164.81, 246.94), (246.94, 329.63, 493.88, 392.00)),
    ],
    "tragic": [  # Am -> F -> C -> G slow emotional lament
        (55.00, (130.81, 164.81, 220.00), (329.63, 440.00, 523.25, 440.00)),
        (87.31, (130.81, 174.61, 220.00), (261.63, 349.23, 440.00, 349.23)),
        (65.41, (130.81, 164.81, 261.63), (329.63, 392.00, 523.25, 392.00)),
        (98.00, (123.47, 146.83, 196.00), (246.94, 293.66, 392.00, 293.66)),
    ],
}

# (step_sec, chime_amp, decay_rate, bass_pulse_hz, perc_kind)
_MOOD_GROOVE: dict[str, tuple[float, float, float, float, str]] = {
    "warm": (0.75, 0.085, 3.8, 0.0, "none"),
    "calm": (0.85, 0.065, 3.2, 0.0, "none"),
    "comedic": (0.375, 0.105, 8.5, 1.33, "pizz"),
    "mysterious": (0.50, 0.085, 5.2, 0.5, "tick"),
    "tense": (0.375, 0.090, 6.2, 1.5, "tick"),
    "action": (0.25, 0.115, 7.0, 2.0, "kick"),
    "tragic": (1.00, 0.075, 2.8, 0.0, "none"),
}


def _load_external_mood_clip(
    custom_bgm_dir: Path | None,
    mood: str,
    target_samples: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Optionally decode user-provided assets/bgm/<mood>.(wav|mp3|ogg|flac|m4a) via ffmpeg."""
    if not custom_bgm_dir or not custom_bgm_dir.exists():
        return None
    cand: Path | None = None
    for ext in ("wav", "mp3", "ogg", "flac", "m4a"):
        p = custom_bgm_dir / f"{mood}.{ext}"
        if p.exists() and p.is_file() and p.stat().st_size > 1024:
            cand = p
            break
    if cand is None:
        return None
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [
                ff,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(cand),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "2",
                "-ar",
                str(sample_rate),
                "-",
            ],
            capture_output=True,
            timeout=20,
        )
        if proc.returncode != 0 or len(proc.stdout) < 8:
            return None
        pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64) / 32768.0
        n_frames = len(pcm) // 2
        if n_frames < 128:
            return None
        stereo = pcm[: n_frames * 2].reshape(n_frames, 2)
        if n_frames < target_samples:
            reps = (target_samples // n_frames) + 1
            stereo = np.tile(stereo, (reps, 1))
        stereo = stereo[:target_samples]
        return stereo[:, 0], stereo[:, 1]
    except Exception:
        return None


def _render_mood_segment(
    mood: str,
    seg_t: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize musical 4-bar chord progression + groove tailored to mood."""
    chords = _MOOD_PROGRESSIONS.get(mood, _MOOD_PROGRESSIONS["tense"])
    step_dur, chime_amp, decay_rate, pulse_hz, perc_kind = _MOOD_GROOVE.get(
        mood, _MOOD_GROOVE["tense"]
    )
    n = len(seg_t)
    seg_l = np.zeros(n, dtype=np.float64)
    seg_r = np.zeros(n, dtype=np.float64)

    bar_sec = 4.0
    cycle_sec = bar_sec * len(chords)
    rel_t = seg_t - seg_t[0]
    phase = np.mod(rel_t, cycle_sec)

    for c_idx, (bass_hz, pad_freqs, motif_freqs) in enumerate(chords):
        lt = phase - c_idx * bar_sec
        mask = (lt >= 0.0) & (lt < bar_sec)
        if not np.any(mask):
            continue
        t_sub = seg_t[mask]
        lt_sub = lt[mask]

        # Gentle chord envelope so bar boundaries never click
        bar_env = 0.45 + 0.55 * (np.sin(np.pi * lt_sub / bar_sec) ** 0.35)

        # Warm sub + fifth/octave harmonic bass with optional tense pulse
        pulse_mod = 1.0 if pulse_hz <= 0 else (0.72 + 0.28 * np.sin(2.0 * np.pi * pulse_hz * lt_sub))
        bass = pulse_mod * (
            0.24 * np.sin(2.0 * np.pi * bass_hz * t_sub)
            + 0.08 * np.sin(2.0 * np.pi * (bass_hz * 2.0) * t_sub)
        )

        # Warm lush detuned string/pad keeping 1-4kHz vocal clarity range untouched
        pad_l = np.zeros_like(t_sub)
        pad_r = np.zeros_like(t_sub)
        for p_idx, f_hz in enumerate(pad_freqs):
            det = 1.0 + (0.0021 if p_idx % 2 == 0 else -0.0021)
            pad_l += 0.095 * np.sin(2.0 * np.pi * f_hz * t_sub) + 0.025 * np.sin(
                2.0 * np.pi * (f_hz * 2.0) * t_sub
            )
            pad_r += 0.095 * np.sin(2.0 * np.pi * (f_hz * det) * t_sub + 0.35) + 0.025 * np.sin(
                2.0 * np.pi * (f_hz * 2.0 * det) * t_sub + 0.2
            )

        # Electric-piano / celesta harmonic pluck arpeggio (rich harmonics, soft attack)
        step_idx = np.floor(lt_sub / step_dur).astype(np.int64)
        step_t = lt_sub - step_idx * step_dur
        note_f = np.array([motif_freqs[s % len(motif_freqs)] for s in step_idx], dtype=np.float64)
        env = np.exp(-step_t * decay_rate) * np.minimum(1.0, step_t / 0.012)
        key_tone = chime_amp * env * (
            0.78 * np.sin(2.0 * np.pi * note_f * t_sub)
            + 0.18 * np.sin(2.0 * np.pi * (note_f * 2.0) * t_sub)
            + 0.05 * np.sin(2.0 * np.pi * (note_f * 3.0) * t_sub)
        )
        pan_l = np.where((step_idx % 2) == 0, 0.85, 0.58)
        pan_r = np.where((step_idx % 2) == 0, 0.58, 0.85)

        seg_l[mask] = (bass + pad_l) * bar_env + key_tone * pan_l
        seg_r[mask] = (bass + pad_r) * bar_env + key_tone * pan_r

    # Subtle rhythmic percussion layer (clock tick in tense/mystery, low tom/kick in action)
    if perc_kind == "kick":
        beat_step = 0.5
        bt = np.mod(rel_t, beat_step)
        kick_env = np.exp(-bt * 18.0) * np.minimum(1.0, bt / 0.004)
        kick_sig = 0.22 * kick_env * np.sin(2.0 * np.pi * (105.0 - 65.0 * np.minimum(1.0, bt * 6.0)) * bt)
        seg_l += kick_sig
        seg_r += kick_sig
    elif perc_kind == "tick":
        tick_step = 0.75 if mood == "tense" else 1.0
        tt = np.mod(rel_t, tick_step)
        tick_env = np.exp(-tt * 75.0) * np.minimum(1.0, tt / 0.002)
        noise = rng.uniform(-1.0, 1.0, size=n)
        # Soft low-mid clock woodblock tick (never harsh)
        tick_sig = 0.045 * tick_env * (0.65 * np.sin(2.0 * np.pi * 620.0 * tt) + 0.35 * noise)
        seg_l += tick_sig * 0.9
        seg_r += tick_sig * 0.7

    return seg_l, seg_r


def build_dynamic_timeline_bgm(
    timeline,
    output_wav: Path,
    sample_rate: int = 48000,
    custom_bgm_dir: Path | None = None,
) -> Path:
    """Build continuous multi-mood musical BGM (external assets if provided, else rich 4-bar synth) + layered SFX."""
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    dur_sec = max(1.0, timeline.total_duration_ms / 1000.0)
    n_samples = int(round(dur_sec * sample_rate))
    t = np.linspace(0.0, dur_sec, n_samples, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(42)

    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    segments: list[tuple[int, int, str]] = []
    for shot in timeline.shots:
        mood = shot.mood if shot.mood in _MOOD_PROGRESSIONS else "tense"
        if segments and segments[-1][2] == mood:
            s0, _, m0 = segments[-1]
            segments[-1] = (s0, shot.end_ms, m0)
        else:
            segments.append((shot.start_ms, shot.end_ms, mood))
    if not segments:
        segments = [(0, timeline.total_duration_ms, "tense")]

    xf_samples = int(1.2 * sample_rate)
    for idx, (s_ms, e_ms, mood) in enumerate(segments):
        i0 = max(0, int(round(s_ms / 1000.0 * sample_rate)) - (xf_samples // 2 if idx > 0 else 0))
        i1 = min(
            n_samples,
            int(round(e_ms / 1000.0 * sample_rate)) + (xf_samples // 2 if idx < len(segments) - 1 else 0),
        )
        if i1 <= i0:
            continue

        ext_clip = _load_external_mood_clip(custom_bgm_dir, mood, i1 - i0, sample_rate)
        if ext_clip is not None:
            seg_l, seg_r = ext_clip
        else:
            seg_l, seg_r = _render_mood_segment(mood, t[i0:i1], rng)

        w = np.ones(i1 - i0, dtype=np.float64)
        if idx > 0:
            fin = min(xf_samples, len(w) // 2)
            if fin > 0:
                w[:fin] *= 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, fin)))
        if idx < len(segments) - 1:
            fout = min(xf_samples, len(w) // 2)
            if fout > 0:
                w[-fout:] *= 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, fout)))
        left[i0:i1] += seg_l * w
        right[i0:i1] += seg_r * w

    # Layered SFX synchronized to camera impact, flash reveal, and scene transitions
    for shot in timeline.shots:
        s_idx = int(round(shot.start_ms / 1000.0 * sample_rate))
        if shot.camera_move == "shake_impact" and 0 <= s_idx < n_samples:
            slen = min(int(0.42 * sample_rate), n_samples - s_idx)
            if slen > 64:
                st = np.linspace(0.0, slen / sample_rate, slen, endpoint=False)
                sub_hit = 0.36 * np.exp(-st * 9.5) * np.sin(
                    2.0 * np.pi * (110.0 - 75.0 * np.minimum(1.0, st * 4.5)) * st
                )
                crack = 0.14 * np.exp(-st * 45.0) * rng.uniform(-1.0, 1.0, size=slen)
                left[s_idx : s_idx + slen] += sub_hit + crack
                right[s_idx : s_idx + slen] += sub_hit + crack
        elif shot.transition_in == "flash_white" and 0 <= s_idx < n_samples:
            # Reverse-swell noise whoosh leading into shimmering fifth sting
            slen = min(int(0.35 * sample_rate), n_samples - s_idx)
            if slen > 64:
                st = np.linspace(0.0, slen / sample_rate, slen, endpoint=False)
                whoosh_env = np.exp(-st * 11.0) * np.sin(np.pi * np.minimum(1.0, st / 0.14))
                whoosh = 0.11 * whoosh_env * np.convolve(
                    rng.uniform(-1.0, 1.0, size=slen), np.ones(12) / 12.0, mode="same"
                )
                shimmer = 0.08 * np.exp(-st * 8.5) * (
                    np.sin(2.0 * np.pi * 523.25 * st) + 0.6 * np.sin(2.0 * np.pi * 783.99 * st)
                )
                left[s_idx : s_idx + slen] += whoosh + shimmer * 0.85
                right[s_idx : s_idx + slen] += whoosh + shimmer * 1.0
        elif shot.transition_in == "fade_black" and 0 <= s_idx < n_samples:
            # Soft airy scene-transition swish
            slen = min(int(0.40 * sample_rate), n_samples - s_idx)
            if slen > 64:
                st = np.linspace(0.0, slen / sample_rate, slen, endpoint=False)
                swish_env = np.sin(np.pi * st / (slen / sample_rate)) ** 1.4
                raw_noise = rng.uniform(-1.0, 1.0, size=slen)
                smooth_noise = np.convolve(raw_noise, np.ones(24) / 24.0, mode="same")
                swish = 0.12 * swish_env * smooth_noise
                left[s_idx : s_idx + slen] += swish
                right[s_idx : s_idx + slen] += swish

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) or 1.0
    stereo = (stereo / peak) * 0.62
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return output_wav
