"""Tests for CV panel sorting, manhwa slicing, manga YOLO, and SoM rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.cv.manga_detector import MangaYoloDetector, sort_panels_reading_order
from src.cv.manhwa_slicer import ManhwaStripSlicer
from src.cv.som import render_som_page
from src.schemas.ingestion import BBox


def test_rtl_sorting() -> None:
    top_left = BBox(x1=10, y1=10, x2=200, y2=200)
    top_right = BBox(x1=220, y1=15, x2=420, y2=205)
    bottom = BBox(x1=10, y1=230, x2=420, y2=450)
    ordered = sort_panels_reading_order([top_left, bottom, top_right], reading_order="rtl")
    assert ordered == [top_right, top_left, bottom]


def test_manhwa_slicer_splits_gutters(tmp_path: Path) -> None:
    strip = np.full((600, 300, 3), 255, dtype=np.uint8)
    strip[20:200, 20:280] = np.random.randint(0, 200, (180, 260, 3), dtype=np.uint8)
    strip[260:520, 20:280] = np.random.randint(0, 200, (260, 260, 3), dtype=np.uint8)
    img_path = tmp_path / "strip.jpg"
    cv2.imwrite(str(img_path), strip)

    slicer = ManhwaStripSlicer()
    page = slicer.detect_page(img_path, "ch01", 1, tmp_path / "panels")
    assert len(page.panels) == 2
    assert (tmp_path / "panels" / "ch01_p001_P1.jpg").exists()


def test_manga_detector_and_som(tmp_path: Path) -> None:
    img = np.full((600, 400, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (380, 280), (20, 20, 20), 4)
    img_path = tmp_path / "page.jpg"
    cv2.imwrite(str(img_path), img)

    det = MangaYoloDetector()
    page = det.detect_page(img_path, "ch01", 1, tmp_path / "panels")
    assert len(page.panels) >= 1
    som_path = render_som_page(img_path, page.panels, tmp_path / "som.jpg")
    assert som_path.exists()
