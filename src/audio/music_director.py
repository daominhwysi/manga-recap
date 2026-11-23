"""Smart Music Director: genre-palette matching, scene-level cue spotting, LUFS-matched mixing & SFX cooldown."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.audio.music_library import (
    MusicCatalog,
    MusicTrack,
    SfxClip,
    decode_audio_pcm,
    load_music_catalog,
)
from src.schemas.script import VALID_MOODS

# Soft mood compatibility for fallback scoring when a track doesn't match the exact primary mood
_RELATED_MOODS: dict[str, tuple[str, ...]] = {
    "mysterious": ("tense", "calm", "tragic"),
    "tense": ("mysterious", "action", "tragic"),
    "action": ("tense",),
    "tragic": ("mysterious", "calm", "warm"),
    "warm": ("calm", "tragic"),
    "calm": ("warm", "mysterious", "tragic"),
    "comedic": ("warm", "calm"),
}

# General story-tone -> music genre affinity keywords (100% series-agnostic)
_TONE_TO_GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dark_fantasy": ("gothic", "hắc ám", "dark fantasy", "ma thuật", "nguyền", "vực thẳm", "bí ẩn", "siêu nhiên", "huyền ảo", "quái vật", "linh hồn", "dị giới"),
    "mystery": ("bí ẩn", "trinh thám", "âm mưu", "giải mã", "manh mối", "nghi vấn", "mystery", "thriller", "che giấu", "sự thật"),
    "action_epic": ("hành động", "chiến đấu", "kịch tính", "bùng nổ", "sức mạnh", "trận chiến", "action", "shonen", "đối đầu", "quyết đấu"),
    "tragedy": ("bi kịch", "đau thương", "mất mát", "tổn thương", "cô đơn", "phản bội", "nước mắt", "tragic", "u buồn"),
    "horror": ("kinh dị", "rùng rợn", "ám ảnh", "điên loạn", "máu me", "horror", "ma quái"),
    "slice_of_life": ("đời thường", "ấm áp", "học đường", "lãng mạn", "thanh xuân", "nhẹ nhàng", "chữa lành", "slice of life", "romance"),
    "comedy": ("hài hước", "dí dỏm", "tấu hài", "vui nhộn", "comedy", "nghịch ngợm"),
    "sci_fi": ("khoa học", "tương lai", "công nghệ", "robot", "không gian", "sci-fi", "cyber"),
}


class MusicCue(BaseModel):
    cue_id: str
    start_ms: int
    end_ms: int
    mood: str
    track_id: str = ""
    track_title: str = ""
    track_artist: str = ""
    track_license: str = ""
    track_file: str = ""
    start_offset_sec: float = 0.0
    score: float = 0.0


def infer_genre_preferences(tone_summary: str = "", title: str = "") -> dict[str, float]:
    """Infer soft genre affinity weights (0.0..1.0) from StoryBible tone_summary & title."""
    blob = f"{tone_summary} {title}".lower()
    raw_scores: dict[str, float] = {}
    for genre, kws in _TONE_TO_GENRE_KEYWORDS.items():
        hits = sum(1.0 for kw in kws if kw in blob)
        if hits > 0:
            raw_scores[genre] = hits
    if not raw_scores:
        return {"general": 1.0}
    max_v = max(raw_scores.values()) or 1.0
    return {g: round(v / max_v, 3) for g, v in raw_scores.items()}


def plan_music_cues(
    timeline,
    min_cue_duration_ms: int = 28000,
) -> list[MusicCue]:
    """Group timeline shots into cohesive scene-level music cues (avoiding frantic 5s track switches)."""
    if not timeline.shots:
        return [
            MusicCue(
                cue_id="cue_01",
                start_ms=0,
                end_ms=max(1000, timeline.total_duration_ms),
                mood="mysterious",
            )
        ]

    # Step 1: Collapse consecutive shots sharing the same mood
    raw_spans: list[dict[str, Any]] = []
    for shot in timeline.shots:
        mood = shot.mood if shot.mood in VALID_MOODS else "tense"
        hard_break = shot.transition_in == "fade_black"
        if raw_spans and raw_spans[-1]["mood"] == mood and not hard_break:
            raw_spans[-1]["end_ms"] = max(raw_spans[-1]["end_ms"], shot.end_ms)
        else:
            raw_spans.append(
                {
                    "start_ms": shot.start_ms,
                    "end_ms": shot.end_ms,
                    "mood": mood,
                    "hard_break": hard_break,
                }
            )

    # Step 2: Merge overly short mood flickers into neighboring cues
    # Keep high-contrast climactic cues (action/tragic) down to 55% of min_cue if preceded by hard break
    def _effective_min(span: dict[str, Any]) -> int:
        if span["hard_break"] and span["mood"] in ("action", "tragic", "comedic"):
            return int(min_cue_duration_ms * 0.55)
        return min_cue_duration_ms

    spans = list(raw_spans)
    while len(spans) > 1:
        # Find shortest span that violates its effective minimum duration
        violators = [
            (i, s["end_ms"] - s["start_ms"])
            for i, s in enumerate(spans)
            if (s["end_ms"] - s["start_ms"]) < _effective_min(s)
        ]
        if not violators:
            break
        # Merge the shortest span first
        idx = min(violators, key=lambda kv: kv[1])[0]
        cur = spans[idx]

        # Choose neighbor to absorb cur: prefer same-mood, else related-mood, else longer neighbor
        candidates: list[tuple[float, int]] = []
        for nb_i in (idx - 1, idx + 1):
            if 0 <= nb_i < len(spans):
                nb = spans[nb_i]
                nb_dur = nb["end_ms"] - nb["start_ms"]
                aff = 3.0 if nb["mood"] == cur["mood"] else (
                    1.5 if cur["mood"] in _RELATED_MOODS.get(nb["mood"], ()) else 0.0
                )
                # Slightly prefer absorbing into previous cue so transitions stay anchored at scene starts
                dir_bonus = 0.2 if nb_i < idx else 0.0
                candidates.append((aff + dir_bonus + min(1.0, nb_dur / 60000.0), nb_i))

        target_i = max(candidates, key=lambda kv: kv[0])[1]
        if target_i < idx:
            spans[target_i]["end_ms"] = cur["end_ms"]
        else:
            spans[target_i]["start_ms"] = cur["start_ms"]
            spans[target_i]["hard_break"] = spans[target_i]["hard_break"] or cur["hard_break"]
        spans.pop(idx)

        # Collapse any adjacent same-mood spans created by the merge
        collapsed: list[dict[str, Any]] = []
        for s in spans:
            if collapsed and collapsed[-1]["mood"] == s["mood"]:
                collapsed[-1]["end_ms"] = max(collapsed[-1]["end_ms"], s["end_ms"])
            else:
                collapsed.append(s)
        spans = collapsed

    # Ensure cues cleanly span 0 .. total_duration_ms with zero gaps
    spans[0]["start_ms"] = 0
    spans[-1]["end_ms"] = max(spans[-1]["end_ms"], timeline.total_duration_ms)
    for k in range(1, len(spans)):
        spans[k]["start_ms"] = spans[k - 1]["end_ms"]

    return [
        MusicCue(
            cue_id=f"cue_{i + 1:02d}",
            start_ms=int(s["start_ms"]),
            end_ms=int(s["end_ms"]),
            mood=str(s["mood"]),
        )
        for i, s in enumerate(spans)
    ]


def select_tracks_for_cues(
    cues: list[MusicCue],
    catalog: MusicCatalog,
    genre_prefs: dict[str, float] | None = None,
) -> list[MusicCue]:
    """Assign optimal, non-repeating catalog tracks to each cue based on mood, genre palette & energy."""
    if not catalog.tracks:
        return cues

    prefs = genre_prefs or {"general": 1.0}
    target_energy = {
        "calm": 1.8,
        "warm": 2.3,
        "tragic": 2.2,
        "mysterious": 2.8,
        "comedic": 3.2,
        "tense": 3.6,
        "action": 4.6,
    }
    used_counts: dict[str, int] = {}
    prev_track_id = ""
    planned: list[MusicCue] = []

    for cue in cues:
        best_track: MusicTrack | None = None
        best_score = -1e9
        t_eng = target_energy.get(cue.mood, 3.0)

        for tr in catalog.tracks:
            sc = 0.0
            if tr.moods and tr.moods[0] == cue.mood:
                sc += 4.5
            elif cue.mood in tr.moods:
                sc += 2.8
            elif any(m in _RELATED_MOODS.get(cue.mood, ()) for m in tr.moods):
                sc += 1.0

            # Genre palette affinity
            g_hits = [prefs.get(g, 0.0) for g in tr.genres]
            if g_hits:
                sc += 1.8 * max(g_hits)

            # Energy fit
            sc += max(0.0, 1.0 - abs(tr.energy - t_eng) * 0.35)

            # Anti-repeat penalties (strong penalty for immediate consecutive reuse)
            if tr.track_id == prev_track_id:
                sc -= 4.2
            sc -= used_counts.get(tr.track_id, 0) * 1.4

            # Slightly prefer tracks longer than cue duration to minimize looping
            cue_sec = (cue.end_ms - cue.start_ms) / 1000.0
            if tr.duration_sec >= cue_sec:
                sc += 0.35

            if sc > best_score:
                best_score = sc
                best_track = tr

        c_copy = cue.model_copy(deep=True)
        if best_track is not None:
            c_copy.track_id = best_track.track_id
            c_copy.track_title = best_track.title
            c_copy.track_artist = best_track.artist
            c_copy.track_license = best_track.license
            c_copy.track_file = best_track.file_path
            c_copy.start_offset_sec = best_track.active_start_sec
            c_copy.score = round(best_score, 3)
            used_counts[best_track.track_id] = used_counts.get(best_track.track_id, 0) + 1
            prev_track_id = best_track.track_id
        planned.append(c_copy)

    return planned


def _tile_with_loop_crossfade(
    stereo: np.ndarray,
    target_samples: int,
    sample_rate: int,
    loop_xf_sec: float = 2.0,
) -> np.ndarray:
    """Loop a stereo track smoothly to target_samples using equal-power loop crossfade (zero clicks)."""
    n_src = len(stereo)
    if n_src >= target_samples:
        return stereo[:target_samples].copy()

    xf = min(int(loop_xf_sec * sample_rate), n_src // 4)
    if xf < 64:
        reps = (target_samples // n_src) + 1
        return np.tile(stereo, (reps, 1))[:target_samples]

    step = n_src - xf
    out = np.zeros((target_samples, 2), dtype=np.float64)
    pos = 0
    first = True
    t_lin = np.linspace(0.0, 1.0, xf, dtype=np.float64)
    fade_in = np.sin(0.5 * np.pi * t_lin)[:, None]
    fade_out = np.cos(0.5 * np.pi * t_lin)[:, None]

    while pos < target_samples:
        take = min(n_src, target_samples - pos)
        chunk = stereo[:take].copy()
        if not first and take >= xf:
            chunk[:xf] *= fade_in
            out[pos : pos + xf] *= fade_out
        out[pos : pos + take] += chunk
        pos += step
        first = False
    return out


def _overlay_sfx_events(
    timeline,
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    library_dir: Path | None,
    catalog: MusicCatalog | None,
    cooldown_ms: int = 6000,
) -> None:
    """Overlay real/catalog SFX clips at key camera/transition moments with anti-spam cooldown."""
    n_samples = len(left)
    sfx_cache: dict[str, np.ndarray] = {}

    def _get_sfx(category: str) -> np.ndarray | None:
        if category in sfx_cache:
            return sfx_cache[category]
        if library_dir and catalog and catalog.sfx:
            cands = [c for c in catalog.sfx if c.category == category]
            if cands:
                arr = decode_audio_pcm(library_dir / cands[0].file_path, sample_rate=sample_rate, channels=2)
                if arr is not None:
                    sfx_cache[category] = arr
                    return arr
        return None

    last_sfx_ms = -cooldown_ms
    for shot in timeline.shots:
        if (shot.start_ms - last_sfx_ms) < cooldown_ms:
            continue
        cat: str | None = None
        gain = 0.28
        if shot.camera_move == "shake_impact":
            cat = "impact"
            gain = 0.34
        elif shot.transition_in == "flash_white":
            cat = "whoosh"
            gain = 0.26
        elif shot.transition_in == "fade_black":
            cat = "transition"
            gain = 0.22
        if not cat:
            continue

        arr = _get_sfx(cat)
        if arr is None:
            continue
        s_idx = int(round(shot.start_ms / 1000.0 * sample_rate))
        if s_idx < 0 or s_idx >= n_samples:
            continue
        slen = min(len(arr), n_samples - s_idx)
        if slen < 64:
            continue
        left[s_idx : s_idx + slen] += arr[:slen, 0] * gain
        right[s_idx : s_idx + slen] += arr[:slen, 1] * gain
        last_sfx_ms = shot.start_ms


def write_music_credits(cues: list[MusicCue], credits_path: Path) -> None:
    """Write YouTube-ready CC-BY music attribution file for all tracks used in this video."""
    credits_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines = ["# Music Credits (Auto-generated by Manga Recap Generator)", ""]
    for c in cues:
        if not c.track_title or c.track_id in seen:
            continue
        seen.add(c.track_id)
        lines.append(f"- \"{c.track_title}\" by {c.track_artist} — License: {c.track_license}")
    if len(lines) == 2:
        lines.append("- Built-in procedural ambient score (Royalty-Free)")
    credits_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mix_directed_bgm(
    timeline,
    output_wav: Path,
    library_dir: Path | None = None,
    tone_summary: str = "",
    title: str = "",
    min_cue_duration_ms: int = 28000,
    target_lufs: float = -16.0,
    crossfade_sec: float = 2.5,
    sample_rate: int = 48000,
    credits_path: Path | None = None,
    cues_json_path: Path | None = None,
) -> Path | None:
    """Render full-length BGM track from catalog.json using scene-level cues, LUFS matching & equal-power crossfade.

    Returns output_wav if catalog tracks were mixed, or None if no music catalog is available (caller falls back).
    """
    catalog = load_music_catalog(library_dir)
    if catalog is None or not catalog.tracks or library_dir is None:
        return None

    lib_root = Path(library_dir)
    genre_prefs = infer_genre_preferences(tone_summary=tone_summary, title=title)
    raw_cues = plan_music_cues(timeline, min_cue_duration_ms=min_cue_duration_ms)
    planned_cues = select_tracks_for_cues(raw_cues, catalog, genre_prefs=genre_prefs)
    track_lookup = {t.track_id: t for t in catalog.tracks}

    dur_sec = max(1.0, timeline.total_duration_ms / 1000.0)
    n_samples = int(round(dur_sec * sample_rate))
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)
    xf_samples = int(crossfade_sec * sample_rate)

    mixed_any = False
    for idx, cue in enumerate(planned_cues):
        tr = track_lookup.get(cue.track_id)
        if tr is None:
            continue
        track_path = lib_root / tr.file_path
        i0 = max(0, int(round(cue.start_ms / 1000.0 * sample_rate)) - (xf_samples // 2 if idx > 0 else 0))
        i1 = min(
            n_samples,
            int(round(cue.end_ms / 1000.0 * sample_rate)) + (xf_samples // 2 if idx < len(planned_cues) - 1 else 0),
        )
        seg_len = i1 - i0
        if seg_len < 64:
            continue

        decoded = decode_audio_pcm(
            track_path,
            sample_rate=sample_rate,
            channels=2,
            start_sec=tr.active_start_sec,
        )
        if decoded is None:
            continue

        looped = _tile_with_loop_crossfade(decoded, seg_len, sample_rate)

        # Loudness normalization to target_lufs (clamped +/-12 dB to prevent noise floor blowup)
        gain_db = max(-12.0, min(12.0, target_lufs - tr.measured_lufs))
        gain_lin = 10.0 ** (gain_db / 20.0)
        seg_l = looped[:, 0] * gain_lin
        seg_r = looped[:, 1] * gain_lin

        # Equal-power crossfade window (sin / cos) at cue boundaries + gentle start/end fade
        w = np.ones(seg_len, dtype=np.float64)
        if idx > 0:
            fin = min(xf_samples, seg_len // 2)
            if fin > 0:
                w[:fin] *= np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, fin))
        else:
            fin = min(int(0.6 * sample_rate), seg_len // 3)
            if fin > 0:
                w[:fin] *= np.linspace(0.0, 1.0, fin)
        if idx < len(planned_cues) - 1:
            fout = min(xf_samples, seg_len // 2)
            if fout > 0:
                w[-fout:] *= np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, fout))
        else:
            fout = min(int(2.0 * sample_rate), seg_len // 3)
            if fout > 0:
                w[-fout:] *= np.linspace(1.0, 0.0, fout)

        left[i0:i1] += seg_l * w
        right[i0:i1] += seg_r * w
        mixed_any = True

    if not mixed_any:
        return None

    # Selective studio SFX overlay with 6s anti-spam cooldown
    _overlay_sfx_events(timeline, left, right, sample_rate, lib_root, catalog, cooldown_ms=6000)

    # Gentle soft-limiting & headroom
    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) or 1.0
    if peak > 0.85:
        stereo = np.tanh(stereo / max(0.85, peak) * 1.05) * 0.85
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    if cues_json_path is not None:
        cues_json_path.parent.mkdir(parents=True, exist_ok=True)
        cues_json_path.write_text(
            json.dumps(
                {"genre_preferences": genre_prefs, "cues": [c.model_dump() for c in planned_cues]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if credits_path is not None:
        write_music_credits(planned_cues, credits_path)

    return output_wav
