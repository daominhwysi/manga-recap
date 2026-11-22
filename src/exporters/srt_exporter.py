"""SRT subtitle exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.exporters.registry import EXPORTER_REGISTRY
from src.render.subtitles import write_srt
from src.schemas.timeline import VideoTimeline


@EXPORTER_REGISTRY.register("srt")
class SrtExporter:
    """Exports timeline subtitles to 06_subtitles.srt."""

    name = "srt"

    def export(self, timeline: VideoTimeline, job_dir: Path, **kwargs: Any) -> Path:
        return write_srt(timeline.subtitles, job_dir / "06_subtitles.srt")
