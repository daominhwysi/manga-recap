"""Microsoft Edge-TTS backend converting MP3 output to WAV via FFmpeg with WordBoundary timestamps."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import edge_tts
import imageio_ffmpeg

from src.audio.base import BaseTTSBackend, estimate_word_timestamps_from_wav, get_wav_duration_ms
from src.audio.registry import TTS_REGISTRY
from src.schemas.timeline import AudioBeatInfo, WordTimestamp


@TTS_REGISTRY.register("edge_tts")
class EdgeTTSBackend(BaseTTSBackend):
    """Cloud neural TTS via edge-tts capturing native WordBoundary offsets."""

    name = "edge_tts"

    def __init__(self, default_voice: str = "vi-VN-NamMinhNeural", sample_rate: int = 48000) -> None:
        self.default_voice = default_voice
        self.sample_rate = sample_rate

    async def _synth_mp3(
        self,
        text: str,
        mp3_path: Path,
        voice: str,
        rate: str,
    ) -> list[WordTimestamp]:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, boundary="WordBoundary")
        word_ts: list[WordTimestamp] = []
        with mp3_path.open("wb") as fp:
            async for chunk in communicate.stream():
                ctype = chunk.get("type")
                if ctype == "audio":
                    fp.write(chunk["data"])
                elif ctype == "WordBoundary":
                    offset_ms = int(round(chunk.get("offset", 0) / 10_000))
                    dur_ms = int(round(chunk.get("duration", 0) / 10_000))
                    w_txt = str(chunk.get("text", "")).strip()
                    if w_txt:
                        word_ts.append(
                            WordTimestamp(
                                word=w_txt,
                                start_ms=max(0, offset_ms),
                                end_ms=max(offset_ms + 25, offset_ms + dur_ms),
                            )
                        )
        return word_ts

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
        rate: str = "+5%",
    ) -> AudioBeatInfo:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chosen_voice = voice or self.default_voice
        mp3_tmp = output_path.with_suffix(".tmp.mp3")

        word_ts = asyncio.run(self._synth_mp3(text, mp3_tmp, chosen_voice, rate))
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(mp3_tmp),
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        mp3_tmp.unlink(missing_ok=True)

        dur_ms = get_wav_duration_ms(output_path)
        if not word_ts:
            word_ts = estimate_word_timestamps_from_wav(output_path, text)
        return AudioBeatInfo(
            beat_id=output_path.stem,
            audio_path=str(output_path.resolve()),
            duration_ms=dur_ms,
            sample_rate=self.sample_rate,
            word_timestamps=word_ts,
        )
