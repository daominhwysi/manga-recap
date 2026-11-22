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


def test_l_shape_xy_cut_sorting() -> None:
    # Tall panel on right side alongside two stacked panels on left side (RTL: right tall -> top-left -> bot-left)
    tall_right = BBox(x1=220, y1=10, x2=420, y2=450)
    top_left = BBox(x1=10, y1=10, x2=200, y2=210)
    bot_left = BBox(x1=10, y1=230, x2=200, y2=450)
    rtl = sort_panels_reading_order([bot_left, tall_right, top_left], reading_order="rtl")
    assert rtl == [tall_right, top_left, bot_left]
    ltr = sort_panels_reading_order([bot_left, tall_right, top_left], reading_order="ltr")
    assert ltr == [top_left, bot_left, tall_right]


def test_manhwa_yolo_postpass_and_stage1_trim_sync(tmp_path: Path) -> None:
    # 1) ManhwaStripSlicer populates focal_points via injected detector
    class _FakeYolo:
        def _run_yolo(self, crop):
            h, w = crop.shape[:2]
            return {
                "face": [(BBox(x1=10, y1=10, x2=min(w, 60), y2=min(h, 60)), 0.9)],
                "body": [],
                "text": [],
            }

    strip = np.full((600, 300, 3), 255, dtype=np.uint8)
    strip[20:240, 20:280] = np.random.randint(0, 200, (220, 260, 3), dtype=np.uint8)
    strip[300:540, 20:280] = np.random.randint(0, 200, (240, 260, 3), dtype=np.uint8)
    img_path = tmp_path / "manhwa.jpg"
    cv2.imwrite(str(img_path), strip)

    slicer = ManhwaStripSlicer(yolo_detector=_FakeYolo())  # type: ignore
    page = slicer.detect_page(img_path, "ch01", 1, tmp_path / "m_panels")
    assert len(page.panels) == 2
    assert page.panels[0].focal_points and page.panels[0].focal_points[0].kind == "face"

    # 2) Stage 1 white border trim shifts focal coordinates accurately
    from src.cv.manga_detector import trim_white_border_with_offset

    inset = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(inset, (300, 400), (540, 640), (30, 30, 30), -1)
    trimmed, ox, oy = trim_white_border_with_offset(inset)
    assert ox > 200 and oy > 300
    assert trimmed.shape[0] < 450 and trimmed.shape[1] < 450
