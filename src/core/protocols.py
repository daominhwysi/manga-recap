"""Protocol interfaces for pluggable pipeline components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.schemas.ingestion import PageInfo
from src.schemas.timeline import AudioBeatInfo, VideoTimeline


@runtime_checkable
class PanelDetector(Protocol):
    """Detects panels and focal regions from a manga or manhwa page."""

    name: str

    def detect_page(
        self,
        image_path: Path,
        chapter_id: str,
        page_index: int,
        panels_dir: Path,
        reading_order: str = "rtl",
    ) -> PageInfo:
        ...


@runtime_checkable
class TTSBackend(Protocol):
    """Synthesizes speech audio for a single script beat."""

    name: str

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
        rate: str = "+0%",
    ) -> AudioBeatInfo:
        ...


@runtime_checkable
class VideoRenderer(Protocol):
    """Renders a VideoTimeline into an output video file."""

    name: str

    def render(
        self,
        timeline: VideoTimeline,
        output_path: Path,
        subtitles_path: Path | None = None,
    ) -> Path:
        ...


@runtime_checkable
class Exporter(Protocol):
    """Exports timeline/subtitles into external editor or preview formats."""

    name: str

    def export(
        self,
        timeline: VideoTimeline,
        job_dir: Path,
        **kwargs: Any,
    ) -> Path:
        ...


@runtime_checkable
class MangaDownloader(Protocol):
    """Downloads and extracts manga chapters/volumes from a source URL or ID."""

    name: str

    def download(
        self,
        source: str,
        output_dir: Path,
        volume_filter: str | None = None,
    ) -> Path:
        ...
