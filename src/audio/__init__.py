"""Audio synthesis stage: TTS backends, registry, and voiceover stitching."""

from src.audio.base import BaseTTSBackend, concat_beat_wavs, get_wav_duration_ms
from src.audio.edge_tts_backend import EdgeTTSBackend
from src.audio.registry import TTS_REGISTRY
from src.audio.stage import run_stage4_audio
from src.audio.vieneu_tts import VieNeuTTSBackend

__all__ = [
    "BaseTTSBackend",
    "EdgeTTSBackend",
    "TTS_REGISTRY",
    "VieNeuTTSBackend",
    "concat_beat_wavs",
    "get_wav_duration_ms",
    "run_stage4_audio",
]
