"""HTML preview exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.exporters.registry import EXPORTER_REGISTRY
from src.render.preview_html import generate_preview_html
from src.schemas.timeline import VideoTimeline


@EXPORTER_REGISTRY.register("html_preview")
class HtmlPreviewExporter:
    """Exports preview.html storyboard."""

    name = "html_preview"

    def export(self, timeline: VideoTimeline, job_dir: Path, **kwargs: Any) -> Path:
        return generate_preview_html(timeline, job_dir / "preview.html")
