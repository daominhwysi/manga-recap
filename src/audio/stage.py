"""Stage 4 orchestrator: synthesize per-beat voiceovers (with pronunciation normalization) and stitch master WAV."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.audio.base import (
    beat_cache_key,
    concat_beat_wavs,
    estimate_word_timestamps_from_wav,
    get_wav_duration_ms,
)
from src.audio.edge_tts_backend import EdgeTTSBackend  # noqa: F401
from src.audio.registry import TTS_REGISTRY
from src.audio.vieneu_tts import VieNeuTTSBackend  # noqa: F401
from src.config import TTSConfig
from src.schemas.script import RecapScript, ScriptBeat
from src.schemas.timeline import AudioBeatInfo, AudioManifest


def run_stage4_audio(
    script: RecapScript,
    job_dir: Path,
    tts_config: TTSConfig,
    force: bool = False,
) -> AudioManifest:
    """Synthesize all script beats and merge into 04_voiceover.wav."""
    manifest_path = job_dir / "04_audio_manifest.json"
    master_wav = job_dir / "04_voiceover.wav"
    if manifest_path.exists() and master_wav.exists() and not force:
        return AudioManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    backend_name = tts_config.backend
    voice = tts_config.vieneu_voice if backend_name == "vieneu" else tts_config.voice
    backend_kwargs: dict = {
        "default_voice": voice,
        "sample_rate": tts_config.sample_rate,
    }
    if backend_name == "vieneu":
        backend_kwargs["fallback_edge_voice"] = tts_config.voice
    backend = TTS_REGISTRY.create(backend_name, **backend_kwargs)

    beats_dir = job_dir / "audio" / "beats"
    beats_dir.mkdir(parents=True, exist_ok=True)

    def _synth_one(beat: ScriptBeat) -> AudioBeatInfo:
        spoken_text = beat.tts_text.strip() or beat.narration_vi
        wav_name = beat_cache_key(
            beat.beat_id,
            spoken_text,
            voice,
            backend_name,
            rate=tts_config.rate,
        )
        wav_path = beats_dir / wav_name
        if wav_path.exists() and not force:
            dur_ms = get_wav_duration_ms(wav_path)
            word_ts = estimate_word_timestamps_from_wav(wav_path, beat.narration_vi)
            return AudioBeatInfo(
                beat_id=beat.beat_id,
                audio_path=str(wav_path.resolve()),
                duration_ms=dur_ms,
                pause_after_ms=beat.pause_after_ms,
                sample_rate=tts_config.sample_rate,
                word_timestamps=word_ts,
            )

        info = backend.synthesize(
            text=spoken_text,
            output_path=wav_path,
            voice=voice,
            rate=tts_config.rate,
        )
        # Re-align timestamps against display narration_vi so subtitles match exact words
        info.word_timestamps = estimate_word_timestamps_from_wav(wav_path, beat.narration_vi)
        info.beat_id = beat.beat_id
        info.pause_after_ms = beat.pause_after_ms
        return info

    # Parallelize cloud Edge-TTS safely; keep local ONNX VieNeu sequential to avoid thread contention
    if backend_name == "edge_tts" and len(script.beats) > 1:
        with ThreadPoolExecutor(max_workers=3) as pool:
            beat_infos = list(pool.map(_synth_one, script.beats))
    else:
        beat_infos = [_synth_one(beat) for beat in script.beats]

    total_ms, timed_beats = concat_beat_wavs(
        beat_infos,
        master_wav,
        target_sr=tts_config.sample_rate,
    )
    manifest = AudioManifest(
        backend=backend_name,
        voice=voice,
        sample_rate=tts_config.sample_rate,
        total_duration_ms=total_ms,
        voiceover_path=str(master_wav.resolve()),
        beats=timed_beats,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
