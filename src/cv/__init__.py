"""Computer Vision stage: ingestion, panel detection, and Set-of-Mark annotation."""

from src.cv.loader import collect_chapters, natural_sort_key
from src.cv.manga_detector import MangaYoloDetector
from src.cv.manhwa_slicer import ManhwaStripSlicer
from src.cv.registry import DETECTOR_REGISTRY
from src.cv.som import render_som_page
from src.cv.stage import run_stage1_ingest

__all__ = [
    "DETECTOR_REGISTRY",
    "MangaYoloDetector",
    "ManhwaStripSlicer",
    "collect_chapters",
    "natural_sort_key",
    "render_som_page",
    "run_stage1_ingest",
]
