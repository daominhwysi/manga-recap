"""Local Vietnamese neural TTS backend using VieNeu."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.audio.base import BaseTTSBackend, estimate_word_timestamps_from_wav, get_wav_duration_ms
from src.audio.registry import TTS_REGISTRY
from src.schemas.timeline import AudioBeatInfo


@TTS_REGISTRY.register("vieneu")
class VieNeuTTSBackend(BaseTTSBackend):
    """Offline Vietnamese TTS via VieNeu package with genuine Edge-TTS fallback."""

    name = "vieneu"

    def __init__(
        self,
        default_voice: str = "Thiện Minh",
        sample_rate: int = 48000,
        fallback_edge_voice: str = "vi-VN-NamMinhNeural",
    ) -> None:
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.fallback_edge_voice = fallback_edge_voice
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            from vieneu import Vieneu  # type: ignore

            self._engine = Vieneu()
        return self._engine

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
        rate: str = "+0%",
    ) -> AudioBeatInfo:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            engine = self._get_engine()
            chosen_voice = voice or self.default_voice
            audio = engine.infer(text=text, voice=chosen_voice)
            arr = np.asarray(audio, dtype=np.float32)
            sr = getattr(engine, "sample_rate", self.sample_rate)
            sf.write(str(output_path), arr, sr, subtype="PCM_16")
            dur_ms = get_wav_duration_ms(output_path)
            word_ts = estimate_word_timestamps_from_wav(output_path, text)
            return AudioBeatInfo(
                beat_id=output_path.stem,
                audio_path=str(output_path.resolve()),
                duration_ms=dur_ms,
                sample_rate=sr,
                word_timestamps=word_ts,
            )
        except Exception:
            from src.audio.edge_tts_backend import EdgeTTSBackend

            edge_fb = EdgeTTSBackend(
                default_voice=self.fallback_edge_voice,
                sample_rate=self.sample_rate,
            )
            return edge_fb.synthesize(text=text, output_path=output_path, voice=self.fallback_edge_voice, rate=rate)
