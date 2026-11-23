"""Automated Music & SFX Dataset Manager: CC-BY starter fetcher, DSP acoustic analyzer & auto-indexer."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.schemas.script import VALID_MOODS

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
INCOMPETECH_CATALOG_URL = "https://incompetech.com/music/royalty-free/pieces.json"
INCOMPETECH_MP3_BASE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"


class MusicTrack(BaseModel):
    track_id: str
    title: str
    artist: str = "Unknown"
    license: str = "Royalty-Free"
    source_url: str = ""
    file_path: str  # Relative path inside music_library_dir
    file_size: int = 0
    file_mtime: float = 0.0
    duration_sec: float = 0.0
    measured_lufs: float = -16.0
    bpm: float = 100.0
    energy: int = 3  # 1 (ambient) .. 5 (high energy)
    brightness: float = 0.5  # 0.0 (dark/warm) .. 1.0 (bright)
    active_start_sec: float = 0.0
    moods: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    instruments: str = ""
    feel_tags: list[str] = Field(default_factory=list)


class SfxClip(BaseModel):
    sfx_id: str
    category: str  # impact | whoosh | transition | sting | riser
    file_path: str
    duration_sec: float = 0.0
    license: str = "CC0"


class MusicCatalog(BaseModel):
    version: str = "1.0"
    tracks: list[MusicTrack] = Field(default_factory=list)
    sfx: list[SfxClip] = Field(default_factory=list)


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "track"


def decode_audio_pcm(
    audio_path: Path,
    sample_rate: int = 48000,
    channels: int = 2,
    start_sec: float = 0.0,
    max_sec: float | None = None,
) -> np.ndarray | None:
    """Decode any ffmpeg-supported audio file into float64 array of shape (n_frames, channels)."""
    if not audio_path.exists() or audio_path.stat().st_size < 512:
        return None
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-hide_banner", "-loglevel", "error"]
        if start_sec > 0.05:
            cmd.extend(["-ss", f"{start_sec:.3f}"])
        cmd.extend(["-i", str(audio_path)])
        if max_sec is not None and max_sec > 0:
            cmd.extend(["-t", f"{max_sec:.3f}"])
        cmd.extend(
            [
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "-",
            ]
        )
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0 or len(proc.stdout) < 8:
            return None
        pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64) / 32768.0
        n_frames = len(pcm) // channels
        if n_frames < 64:
            return None
        return pcm[: n_frames * channels].reshape(n_frames, channels)
    except Exception:
        return None


def analyze_track_acoustics(audio_path: Path, analysis_sr: int = 22050) -> dict[str, float | int]:
    """Pure NumPy DSP analysis: duration, LUFS loudness, active intro offset, BPM, brightness & energy."""
    pcm = decode_audio_pcm(audio_path, sample_rate=analysis_sr, channels=1)
    if pcm is None or len(pcm) < analysis_sr // 4:
        return {
            "duration_sec": 0.0,
            "measured_lufs": -16.0,
            "active_start_sec": 0.0,
            "bpm": 100.0,
            "brightness": 0.5,
            "energy": 3,
        }

    mono = pcm[:, 0]
    duration_sec = len(mono) / float(analysis_sr)

    # Frame RMS over 0.25s hops
    hop = max(256, analysis_sr // 4)
    n_hops = max(1, len(mono) // hop)
    trimmed = mono[: n_hops * hop].reshape(n_hops, hop)
    rms_env = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)

    # Active-region loudness (top 70% loudest windows, approximating gated integrated LUFS)
    nonzero_rms = rms_env[rms_env > 1e-4]
    if len(nonzero_rms) > 0:
        gate_thresh = np.percentile(nonzero_rms, 30)
        gated = nonzero_rms[nonzero_rms >= gate_thresh]
        mean_sq = float(np.mean(gated**2)) if len(gated) > 0 else float(np.mean(nonzero_rms**2))
        measured_lufs = max(-36.0, min(-4.0, 10.0 * math.log10(mean_sq + 1e-12) - 0.691))
    else:
        measured_lufs = -24.0

    # Active start offset: skip slow silent/barely-audible intros (up to first 18s or 15% of track)
    peak_rms = float(np.percentile(rms_env, 90)) if len(rms_env) > 0 else 0.05
    start_thresh = max(0.012, peak_rms * 0.24)
    active_hop = 0
    max_intro_hop = min(len(rms_env) - 1, int(min(18.0, duration_sec * 0.15) * 4))
    for h_idx in range(max(0, max_intro_hop)):
        if rms_env[h_idx] >= start_thresh:
            active_hop = max(0, h_idx - 1)
            break
    active_start_sec = round(active_hop * (hop / float(analysis_sr)), 2)

    # Spectral centroid (brightness 0..1) on representative middle 20s slice
    mid_s = int(min(len(mono) - 1, max(0, (len(mono) // 2) - analysis_sr * 10)))
    mid_e = min(len(mono), mid_s + analysis_sr * 20)
    mid_seg = mono[mid_s:mid_e]
    win_len = 2048
    n_wins = max(1, len(mid_seg) // win_len)
    if n_wins >= 1 and len(mid_seg) >= win_len:
        frames = mid_seg[: n_wins * win_len].reshape(n_wins, win_len) * np.hanning(win_len)
        spec = np.abs(np.fft.rfft(frames, axis=1))
        freqs = np.fft.rfftfreq(win_len, d=1.0 / analysis_sr)
        mag_sum = np.sum(spec, axis=1) + 1e-9
        centroids = np.sum(spec * freqs[None, :], axis=1) / mag_sum
        median_hz = float(np.median(centroids))
        # Map 400Hz..3800Hz -> 0.0..1.0
        brightness = round(max(0.0, min(1.0, (median_hz - 400.0) / 3400.0)), 3)
    else:
        brightness = 0.5

    # Onset strength autocorrelation BPM estimate (65..175 BPM)
    onset_hop = max(128, analysis_sr // 50)  # 50 Hz envelope
    n_oh = len(mid_seg) // onset_hop
    bpm = 100.0
    onset_rate = 0.0
    if n_oh > 120:
        o_frames = mid_seg[: n_oh * onset_hop].reshape(n_oh, onset_hop)
        o_rms = np.sqrt(np.mean(o_frames**2, axis=1) + 1e-12)
        diff = np.maximum(0.0, np.diff(o_rms))
        onset_rate = float(np.mean(diff > (np.mean(diff) + np.std(diff))))
        diff_z = diff - np.mean(diff)
        if np.std(diff_z) > 1e-6:
            ac = np.correlate(diff_z, diff_z, mode="full")[len(diff_z) - 1 :]
            env_sr = analysis_sr / float(onset_hop)
            min_lag = max(1, int(round(env_sr * 60.0 / 175.0)))
            max_lag = min(len(ac) - 1, int(round(env_sr * 60.0 / 65.0)))
            if max_lag > min_lag:
                best_lag = min_lag + int(np.argmax(ac[min_lag : max_lag + 1]))
                bpm = round(60.0 * env_sr / float(best_lag), 1)

    # Energy 1..5 combining loudness dynamics, onset density & tempo
    dyn_score = (measured_lufs + 30.0) / 18.0  # ~0..1
    tempo_score = (bpm - 65.0) / 100.0
    raw_e = 0.45 * max(0.0, min(1.0, dyn_score)) + 0.30 * max(0.0, min(1.0, tempo_score)) + 0.25 * min(1.0, onset_rate * 4.5)
    energy = max(1, min(5, int(round(1 + raw_e * 4))))

    return {
        "duration_sec": round(duration_sec, 2),
        "measured_lufs": round(measured_lufs, 2),
        "active_start_sec": active_start_sec,
        "bpm": bpm,
        "brightness": brightness,
        "energy": energy,
    }


_MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "action": ("action", "epic", "aggressive", "battle", "chase", "driving", "intense", "heroic", "combat", "fast"),
    "tense": ("tense", "suspense", "suspenseful", "unnerving", "thriller", "dark", "ominous", "danger", "horror", "eerie"),
    "mysterious": ("mysterious", "mystery", "mystical", "eerie", "curious", "clockwork", "gothic", "arcane", "secret", "stealth"),
    "tragic": ("tragic", "somber", "sad", "melanchol", "sorrow", "lament", "grief", "emotional", "lonely", "heartbreak"),
    "warm": ("warm", "uplifting", "bright", "hopeful", "romantic", "heartfelt", "gentle", "nostalg", "tender", "sweet"),
    "calm": ("calm", "calming", "relaxed", "ambient", "peaceful", "lofi", "lo-fi", "chill", "soft", "zen", "meditat"),
    "comedic": ("comedic", "comedy", "humorous", "bouncy", "playful", "quirky", "funny", "whimsical", "jaunty", "silly"),
}

_GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dark_fantasy": ("gothic", "mystical", "dark", "choir", "harpsichord", "organ", "cello", "strings", "fantasy", "magic", "medieval", "clock"),
    "mystery": ("mysterious", "mystery", "suspense", "detective", "stealth", "eerie", "piano", "marimba", "pizzicato"),
    "action_epic": ("epic", "action", "battle", "orchestr", "brass", "taiko", "drums", "percussion", "aggressive", "driving"),
    "tragedy": ("somber", "sad", "tragic", "lament", "violin", "cello", "piano", "melanchol", "emotional"),
    "horror": ("horror", "unnerving", "creepy", "haunt", "dread", "eerie", "nightmare"),
    "slice_of_life": ("relaxed", "calming", "warm", "acoustic", "guitar", "lofi", "bright", "uplifting", "flute", "gentle"),
    "comedy": ("humorous", "bouncy", "playful", "quirky", "pizzicato", "comedy", "jaunty", "whimsical"),
    "sci_fi": ("synth", "electronic", "cyber", "space", "futuristic", "sci-fi", "techno"),
}


def classify_track_tags(
    title: str,
    rel_path: str,
    feel_tags: list[str],
    instruments: str,
    description: str,
    acoustics: dict[str, float | int],
) -> tuple[list[str], list[str]]:
    """Infer canonical moods (subset of VALID_MOODS) and genre tags from metadata + DSP acoustics."""
    blob = " ".join([title, rel_path.replace("/", " ").replace("_", " "), " ".join(feel_tags), instruments, description]).lower()

    mood_scores: dict[str, float] = {m: 0.0 for m in VALID_MOODS}
    # Boost if parent folder explicitly names a canonical mood
    parts = [p.lower() for p in Path(rel_path).parts[:-1]]
    for m in VALID_MOODS:
        if m in parts:
            mood_scores[m] += 3.5

    for m, kws in _MOOD_KEYWORDS.items():
        for kw in kws:
            if kw in blob:
                mood_scores[m] += 1.4

    # DSP acoustic priors (ensure even completely untagged MP3 files get accurate mood tags!)
    bpm = float(acoustics.get("bpm", 100.0))
    energy = int(acoustics.get("energy", 3))
    brightness = float(acoustics.get("brightness", 0.5))

    if energy >= 4 and bpm >= 115:
        mood_scores["action"] += 1.2
    if energy >= 3 and brightness < 0.42:
        mood_scores["tense"] += 1.0
    if energy <= 3 and brightness < 0.45:
        mood_scores["mysterious"] += 0.9
    if energy <= 2 and bpm <= 95 and brightness < 0.48:
        mood_scores["tragic"] += 1.0
    if energy <= 2 and brightness >= 0.42:
        mood_scores["calm"] += 1.0
        mood_scores["warm"] += 0.8
    if brightness >= 0.55 and 95 <= bpm <= 135 and energy >= 3:
        mood_scores["comedic"] += 0.8

    ranked_moods = [m for m, sc in sorted(mood_scores.items(), key=lambda kv: -kv[1]) if sc >= 1.0][:2]
    if not ranked_moods:
        ranked_moods = [max(mood_scores.items(), key=lambda kv: kv[1])[0]]

    genre_scores: dict[str, float] = {}
    for g, kws in _GENRE_KEYWORDS.items():
        sc = sum(1.0 for kw in kws if kw in blob)
        if sc > 0:
            genre_scores[g] = sc
    ranked_genres = [g for g, _ in sorted(genre_scores.items(), key=lambda kv: -kv[1])[:3]]
    if not ranked_genres:
        ranked_genres = ["general"]

    return ranked_moods, ranked_genres


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, sample_rate: int = 48000) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) or 1.0
    if peak > 0.92:
        stereo = stereo * (0.92 / peak)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return round(len(left) / float(sample_rate), 3)


def ensure_starter_sfx(library_dir: Path, sample_rate: int = 48000) -> list[SfxClip]:
    """Generate clean, non-harsh studio SFX WAV clips into library_dir/sfx/ if missing, and index all SFX."""
    sfx_dir = library_dir / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(101)

    # 1) Deep cinematic sub-impact boom (for shake_impact)
    impact_p = sfx_dir / "impact_sub_boom.wav"
    if not impact_p.exists():
        n = int(0.65 * sample_rate)
        st = np.linspace(0.0, n / sample_rate, n, endpoint=False)
        freq = 95.0 * np.exp(-st * 5.2) + 32.0
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        sub = 0.65 * np.exp(-st * 6.0) * np.minimum(1.0, st / 0.005) * np.sin(phase)
        punch = 0.25 * np.exp(-st * 38.0) * np.convolve(rng.uniform(-1.0, 1.0, size=n), np.ones(6) / 6.0, mode="same")
        sig = np.tanh((sub + punch) * 1.15)
        _write_stereo_wav(impact_p, sig, sig * 0.98, sample_rate)

    # 2) Airy reverse-swell whoosh + harmonic shimmer (for flash_white reveals)
    whoosh_p = sfx_dir / "whoosh_flash_sting.wav"
    if not whoosh_p.exists():
        n = int(0.55 * sample_rate)
        st = np.linspace(0.0, n / sample_rate, n, endpoint=False)
        env = np.exp(-np.abs(st - 0.16) * 12.0)
        noise = np.convolve(rng.uniform(-1.0, 1.0, size=n), np.ones(18) / 18.0, mode="same")
        shimmer = 0.22 * np.exp(-np.maximum(0.0, st - 0.14) * 7.5) * (
            np.sin(2.0 * np.pi * 523.25 * st) + 0.55 * np.sin(2.0 * np.pi * 783.99 * st)
        )
        sig_l = 0.45 * env * noise + shimmer * 0.88
        sig_r = 0.45 * env * np.roll(noise, 14) + shimmer * 1.0
        _write_stereo_wav(whoosh_p, sig_l, sig_r, sample_rate)

    # 3) Soft airy scene transition swish (for fade_black)
    trans_p = sfx_dir / "transition_soft_swish.wav"
    if not trans_p.exists():
        n = int(0.50 * sample_rate)
        st = np.linspace(0.0, n / sample_rate, n, endpoint=False)
        env = np.sin(np.pi * st / (n / sample_rate)) ** 1.6
        noise = np.convolve(rng.uniform(-1.0, 1.0, size=n), np.ones(36) / 36.0, mode="same")
        low_bed = 0.14 * env * np.sin(2.0 * np.pi * 110.0 * st)
        sig_l = 0.35 * env * noise + low_bed
        sig_r = 0.35 * env * np.roll(noise, 22) + low_bed
        _write_stereo_wav(trans_p, sig_l, sig_r, sample_rate)

    clips: list[SfxClip] = []
    for f in sorted(sfx_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        low = f.stem.lower()
        if any(k in low for k in ("impact", "boom", "hit", "punch", "thud", "slam")):
            cat = "impact"
        elif any(k in low for k in ("whoosh", "flash", "sting", "reveal", "shine")):
            cat = "whoosh"
        elif any(k in low for k in ("riser", "swell", "build")):
            cat = "riser"
        else:
            cat = "transition"
        pcm = decode_audio_pcm(f, sample_rate=16000, channels=1)
        dur = round(len(pcm) / 16000.0, 3) if pcm is not None else 0.5
        clips.append(
            SfxClip(
                sfx_id=_slugify(f.stem),
                category=cat,
                file_path=str(f.relative_to(library_dir).as_posix()),
                duration_sec=dur,
                license="CC0 / Custom",
            )
        )
    return clips


def index_music_library(
    library_dir: Path,
    import_dir: Path | None = None,
    metadata_hints: dict[str, dict[str, Any]] | None = None,
    force_reanalyze: bool = False,
) -> MusicCatalog:
    """Scan library_dir/bgm/ (and optionally import custom audio folder), analyze DSP acoustics & write catalog.json."""
    library_dir = Path(library_dir)
    bgm_dir = library_dir / "bgm"
    bgm_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = library_dir / "catalog.json"

    # Optionally import user audio files preserving relative subfolder hints (e.g. custom/action/x.mp3)
    if import_dir is not None and Path(import_dir).exists():
        src_root = Path(import_dir).resolve()
        for src_f in sorted(src_root.rglob("*")):
            if not src_f.is_file() or src_f.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            rel = src_f.relative_to(src_root)
            dst = bgm_dir / "custom" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_size != src_f.stat().st_size:
                shutil.copy2(src_f, dst)

    existing_by_path: dict[str, MusicTrack] = {}
    if catalog_path.exists() and not force_reanalyze:
        try:
            old_cat = MusicCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
            existing_by_path = {t.file_path: t for t in old_cat.tracks}
        except Exception:
            existing_by_path = {}

    hints = metadata_hints or {}
    tracks: list[MusicTrack] = []
    audio_files = [
        f
        for f in sorted(bgm_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS and f.stat().st_size > 1024
    ]

    for f in audio_files:
        rel_path = f.relative_to(library_dir).as_posix()
        st = f.stat()
        prev = existing_by_path.get(rel_path)
        hint = hints.get(rel_path) or hints.get(f.name) or {}

        if (
            prev is not None
            and not force_reanalyze
            and prev.file_size == st.st_size
            and abs(prev.file_mtime - st.st_mtime) < 1.0
            and not hint
        ):
            tracks.append(prev)
            continue

        acoustics = analyze_track_acoustics(f)
        if acoustics["duration_sec"] < 4.0:
            continue

        title = str(hint.get("title") or f.stem.replace("_", " ").replace("-", " ").strip())
        artist = str(hint.get("artist") or "Custom / Royalty-Free")
        lic = str(hint.get("license") or "Royalty-Free")
        src_url = str(hint.get("source_url") or "")
        instruments = str(hint.get("instruments") or "")
        description = str(hint.get("description") or "")
        feel_tags = [
            x.strip()
            for x in (hint.get("feel") if isinstance(hint.get("feel"), list) else str(hint.get("feel") or "").split(","))
            if x.strip()
        ]

        # Prefer authoritative catalog BPM if numeric and valid
        hint_bpm = 0.0
        try:
            hint_bpm = float(str(hint.get("bpm") or "0").strip())
        except Exception:
            hint_bpm = 0.0
        final_bpm = hint_bpm if 45.0 <= hint_bpm <= 220.0 else float(acoustics["bpm"])

        moods, genres = classify_track_tags(
            title=title,
            rel_path=rel_path,
            feel_tags=feel_tags,
            instruments=instruments,
            description=description,
            acoustics={**acoustics, "bpm": final_bpm},
        )
        # Ensure explicitly requested mood bucket stays primary if hint specifies it
        if hint.get("primary_mood") in VALID_MOODS:
            pm = str(hint["primary_mood"])
            moods = [pm] + [m for m in moods if m != pm]

        tracks.append(
            MusicTrack(
                track_id=_slugify(f"{Path(rel_path).parent.name}_{f.stem}"),
                title=title,
                artist=artist,
                license=lic,
                source_url=src_url,
                file_path=rel_path,
                file_size=st.st_size,
                file_mtime=st.st_mtime,
                duration_sec=float(acoustics["duration_sec"]),
                measured_lufs=float(acoustics["measured_lufs"]),
                bpm=round(final_bpm, 1),
                energy=int(acoustics["energy"]),
                brightness=float(acoustics["brightness"]),
                active_start_sec=float(acoustics["active_start_sec"]),
                moods=moods[:2],
                genres=genres,
                instruments=instruments,
                feel_tags=feel_tags,
            )
        )

    sfx_clips = ensure_starter_sfx(library_dir)
    catalog = MusicCatalog(version="1.0", tracks=tracks, sfx=sfx_clips)
    catalog_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
    return catalog


def load_music_catalog(library_dir: Path | None) -> MusicCatalog | None:
    """Load catalog.json from library_dir if available and non-empty."""
    if library_dir is None:
        return None
    cat_path = Path(library_dir) / "catalog.json"
    if not cat_path.exists():
        return None
    try:
        cat = MusicCatalog.model_validate_json(cat_path.read_text(encoding="utf-8"))
        return cat if (cat.tracks or cat.sfx) else None
    except Exception:
        return None


def _parse_hhmmss(raw: str) -> int:
    parts = [int(p) for p in re.findall(r"\d+", str(raw or ""))]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0


# Dynamic scoring rules to pick diverse, cinematic, voiceover-friendly CC-BY tracks per mood
_FETCH_MOOD_RECIPES: dict[str, dict[str, Any]] = {
    "mysterious": {
        "feel_pos": ("mysterious", "mystical", "eerie", "dark"),
        "feel_neg": ("humorous", "bouncy", "aggressive", "bright"),
        "inst_pos": ("piano", "strings", "harp", "celesta", "marimba", "choir", "synth"),
    },
    "tense": {
        "feel_pos": ("suspenseful", "unnerving", "dark", "intense", "eerie"),
        "feel_neg": ("humorous", "bright", "uplifting", "relaxed", "bouncy"),
        "inst_pos": ("strings", "percussion", "piano", "bass", "cello", "synth"),
    },
    "action": {
        "feel_pos": ("action", "epic", "driving", "intense", "aggressive"),
        "feel_neg": ("calming", "relaxed", "humorous", "somber"),
        "inst_pos": ("strings", "percussion", "drums", "brass", "orchestra", "taiko"),
    },
    "tragic": {
        "feel_pos": ("somber", "dark", "calming"),
        "feel_neg": ("humorous", "bouncy", "aggressive", "bright", "action"),
        "inst_pos": ("piano", "cello", "violin", "strings", "harp"),
    },
    "warm": {
        "feel_pos": ("uplifting", "bright", "relaxed", "calming"),
        "feel_neg": ("dark", "unnerving", "eerie", "aggressive", "suspenseful"),
        "inst_pos": ("piano", "guitar", "acoustic", "strings", "flute", "harp"),
    },
    "calm": {
        "feel_pos": ("calming", "relaxed"),
        "feel_neg": ("aggressive", "intense", "action", "unnerving", "humorous"),
        "inst_pos": ("piano", "synth", "guitar", "harp", "flute", "ambient"),
    },
    "comedic": {
        "feel_pos": ("humorous", "bouncy", "bright"),
        "feel_neg": ("dark", "somber", "unnerving", "eerie", "aggressive"),
        "inst_pos": ("pizzicato", "marimba", "woodwind", "bassoon", "clarinet", "guitar", "piano"),
    },
}


def fetch_starter_music_library(
    library_dir: Path,
    per_mood: int = 3,
    min_duration_sec: int = 75,
    max_duration_sec: int = 300,
    timeout_sec: int = 25,
) -> MusicCatalog:
    """Automatically fetch a balanced starter library of CC-BY 4.0 real tracks from Incompetech & index them."""
    library_dir = Path(library_dir)
    bgm_dir = library_dir / "bgm"
    bgm_dir.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(INCOMPETECH_CATALOG_URL, headers={"User-Agent": "MangaRecapGenerator/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        pieces: list[dict[str, Any]] = json.loads(resp.read().decode("utf-8"))

    used_filenames: set[str] = set()
    download_plan: list[tuple[str, dict[str, Any], Path]] = []
    metadata_hints: dict[str, dict[str, Any]] = {}

    for mood in VALID_MOODS:
        recipe = _FETCH_MOOD_RECIPES[mood]
        scored: list[tuple[float, dict[str, Any]]] = []
        for p in pieces:
            fn = str(p.get("filename") or "").strip()
            if not fn.lower().endswith(".mp3") or fn in used_filenames:
                continue
            dur = _parse_hhmmss(p.get("length", ""))
            if not (min_duration_sec <= dur <= max_duration_sec):
                continue
            feel = str(p.get("feel") or "").lower()
            inst = str(p.get("instruments") or "").lower()
            desc = str(p.get("description") or "").lower()
            # Avoid vocal/singing tracks that clash with Vietnamese voiceover
            if any(bad in inst or bad in desc for bad in ("vocal", "singing", "lyrics", "rap", "speech")):
                continue

            pos_hits = sum(2.0 for k in recipe["feel_pos"] if k in feel)
            neg_hits = sum(2.5 for k in recipe["feel_neg"] if k in feel)
            inst_hits = sum(0.8 for k in recipe["inst_pos"] if k in inst)
            if pos_hits <= 0:
                continue
            score = pos_hits + inst_hits - neg_hits
            # Favor tracks around 2-3.5 minutes for clean looping
            if 110 <= dur <= 220:
                score += 0.6
            scored.append((score, p))

        scored.sort(key=lambda kv: (-kv[0], str(kv[1].get("title") or "")))
        picked_for_mood = 0
        seen_inst_sig: set[str] = set()
        for _, p in scored:
            if picked_for_mood >= per_mood:
                break
            fn = str(p["filename"]).strip()
            inst_sig = str(p.get("instruments") or "").lower()[:24]
            # Promote instrument diversity within the same mood bucket
            if picked_for_mood > 0 and inst_sig and inst_sig in seen_inst_sig and len(scored) > per_mood * 3:
                continue
            seen_inst_sig.add(inst_sig)
            used_filenames.add(fn)
            safe_fn = re.sub(r"[^a-zA-Z0-9._-]+", "_", fn)
            dst = bgm_dir / mood / safe_fn
            rel_path = dst.relative_to(library_dir).as_posix()
            download_plan.append((fn, p, dst))
            metadata_hints[rel_path] = {
                "title": p.get("title") or Path(fn).stem,
                "artist": "Kevin MacLeod (incompetech.com)",
                "license": "CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
                "source_url": INCOMPETECH_MP3_BASE + urllib.parse.quote(fn),
                "instruments": p.get("instruments") or "",
                "description": p.get("description") or "",
                "feel": p.get("feel") or "",
                "bpm": p.get("bpm") or "",
                "primary_mood": mood,
            }
            picked_for_mood += 1

    def _download_one(raw_fn: str, dst: Path) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 16384:
            return True
        url = INCOMPETECH_MP3_BASE + urllib.parse.quote(raw_fn)
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "MangaRecapGenerator/1.0"})
            with urllib.request.urlopen(r, timeout=timeout_sec) as resp:
                data = resp.read()
            if len(data) < 16384:
                return False
            dst.write_bytes(data)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_download_one, raw_fn, dst) for raw_fn, _, dst in download_plan]
        for fut in as_completed(futs):
            fut.result()

    return index_music_library(library_dir, metadata_hints=metadata_hints)
