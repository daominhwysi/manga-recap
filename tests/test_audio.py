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

    # 4) Multi-mood dynamic BGM synthesis (chord progressions, fade_black swish, + external mood clip loading)
    custom_dir = tmp_path / "custom_bgm"
    custom_dir.mkdir()
    sf.write(str(custom_dir / "tense.wav"), np.ones((24000, 2), dtype=np.float32) * 0.05, 48000, subtype="PCM_16")

    tl = VideoTimeline(
        job_id="bgm_test",
        width=1280,
        height=720,
        fps=24,
        total_duration_ms=2500,
        voiceover_path="vo.wav",
        shots=[
            TimelineShot(
                shot_id="s1", beat_id="b1", panel_id="p1", panel_path="p1.jpg",
                start_ms=0, end_ms=1000, mood="calm",
            ),
            TimelineShot(
                shot_id="s2", beat_id="b2", panel_id="p2", panel_path="p2.jpg",
                start_ms=1000, end_ms=1800, camera_move="shake_impact", transition_in="flash_white", mood="action",
            ),
            TimelineShot(
                shot_id="s3", beat_id="b3", panel_id="p3", panel_path="p3.jpg",
                start_ms=1800, end_ms=2500, transition_in="fade_black", mood="tense",
            ),
        ],
    )
    bgm_out = build_dynamic_timeline_bgm(tl, tmp_path / "dyn_bgm.wav", custom_bgm_dir=custom_dir)
    assert bgm_out.exists() and bgm_out.stat().st_size > 10000


def test_music_library_indexer_and_smart_music_director(tmp_path: Path) -> None:
    from src.audio.music_director import (
        infer_genre_preferences,
        mix_directed_bgm,
        plan_music_cues,
        select_tracks_for_cues,
    )
    from src.audio.music_library import index_music_library

    lib_dir = tmp_path / "music_lib"
    sr = 22050
    t = np.linspace(0.0, 6.0, sr * 6, endpoint=False)

    # Create synthetic mood tracks in subfolders with different loudness & spectra
    for mood, freq, amp, fn in [
        ("mysterious", 220.0, 0.12, "gothic_clockwork_mystery.wav"),
        ("action", 520.0, 0.35, "epic_battle_drums.wav"),
        ("tragic", 196.0, 0.10, "somber_cello_lament.wav"),
    ]:
        p = lib_dir / "bgm" / mood / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        sig = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(p), np.stack([sig, sig], axis=1), sr, subtype="PCM_16")

    catalog = index_music_library(lib_dir)
    assert len(catalog.tracks) == 3
    assert len(catalog.sfx) >= 3
    assert (lib_dir / "catalog.json").exists()

    # Genre preference inference from Vietnamese dark fantasy tone
    prefs = infer_genre_preferences("Tông màu gothic huyền bí, ma thuật và bi kịch gia tộc.")
    assert prefs.get("dark_fantasy", 0.0) > 0.5
    assert prefs.get("tragedy", 0.0) > 0.0

    # Scene-level cue spotting merges frantic 5-second mood flickers into smooth >= 28s cues
    shots = [
        TimelineShot(shot_id="s1", beat_id="b1", panel_id="p1", panel_path="p1.jpg", start_ms=0, end_ms=20000, mood="mysterious"),
        TimelineShot(shot_id="s2", beat_id="b2", panel_id="p2", panel_path="p2.jpg", start_ms=20000, end_ms=25000, mood="comedic"),  # 5s flicker -> merged!
        TimelineShot(shot_id="s3", beat_id="b3", panel_id="p3", panel_path="p3.jpg", start_ms=25000, end_ms=42000, mood="mysterious"),
        TimelineShot(shot_id="s4", beat_id="b4", panel_id="p4", panel_path="p4.jpg", start_ms=42000, end_ms=76000, camera_move="shake_impact", transition_in="fade_black", mood="action"),
    ]
    tl = VideoTimeline(
        job_id="dir_test",
        width=1280,
        height=720,
        fps=24,
        total_duration_ms=76000,
        voiceover_path="vo.wav",
        shots=shots,
    )
    cues = plan_music_cues(tl, min_cue_duration_ms=28000)
    assert len(cues) == 2
    assert cues[0].mood == "mysterious" and cues[0].end_ms == 42000
    assert cues[1].mood == "action" and cues[1].start_ms == 42000

    planned = select_tracks_for_cues(cues, catalog, genre_prefs=prefs)
    assert planned[0].track_id != planned[1].track_id

    out_wav = tmp_path / "directed.wav"
    credits_p = tmp_path / "music_credits.txt"
    cues_p = tmp_path / "music_cues.json"
    res = mix_directed_bgm(
        tl,
        out_wav,
        library_dir=lib_dir,
        tone_summary="Tông màu gothic huyền bí và chiến đấu kịch tính.",
        min_cue_duration_ms=28000,
        credits_path=credits_p,
        cues_json_path=cues_p,
    )
    assert res is not None and out_wav.exists() and out_wav.stat().st_size > 100000
    assert credits_p.exists() and "Music Credits" in credits_p.read_text(encoding="utf-8")
    assert cues_p.exists()
