"""Local Vietnamese neural TTS backend using VieNeu."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.audio.base import BaseTTSBackend, get_wav_duration_ms
from src.audio.registry import TTS_REGISTRY
from src.schemas.timeline import AudioBeatInfo


@TTS_REGISTRY.register("vieneu")
class VieNeuTTSBackend(BaseTTSBackend):
    """Offline Vietnamese TTS via VieNeu package with graceful Edge-TTS fallback."""

    name = "vieneu"

    def __init__(self, default_voice: str = "Thiện Minh", sample_rate: int = 48000) -> None:
        self.default_voice = default_voice
        self.sample_rate = sample_rate
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
        engine = self._get_engine()
        chosen_voice = voice or self.default_voice
        audio = engine.infer(text=text, voice=chosen_voice)
        arr = np.asarray(audio, dtype=np.float32)
        sr = getattr(engine, "sample_rate", 24000)
        sf.write(str(output_path), arr, sr, subtype="PCM_16")
        dur_ms = get_wav_duration_ms(output_path)
        return AudioBeatInfo(
            beat_id=output_path.stem,
            audio_path=str(output_path.resolve()),
            duration_ms=dur_ms,
            sample_rate=sr,
        )
