"""Registry for TTS backends."""

from src.core.protocols import TTSBackend
from src.core.registry import Registry

TTS_REGISTRY: Registry[TTSBackend] = Registry("tts_backend")
