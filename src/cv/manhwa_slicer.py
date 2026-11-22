"""Manhwa/Webtoon vertical strip slicer using uniform gutter detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.cv.registry import DETECTOR_REGISTRY
from src.schemas.ingestion import BBox, PageInfo, PanelInfo


@DETECTOR_REGISTRY.register("manhwa_slicer")
class ManhwaStripSlicer:
    """Splits tall vertical webtoon strips into clean individual panels."""

    name = "manhwa_slicer"

    def __init__(self, min_gutter_px: int = 18, min_panel_height: int = 80) -> None:
        self.min_gutter_px = min_gutter_px
        self.min_panel_height = min_panel_height

    def detect_page(
        self,
        image_path: Path,
        chapter_id: str,
        page_index: int,
        panels_dir: Path,
        reading_order: str = "vertical",
    ) -> PageInfo:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read manhwa image: {image_path}")

        h, w = img.shape[:2]
        page_id = f"{chapter_id}_p{page_index:03d}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Rows with very low std-dev are uniform white/black gutters
        row_std = np.std(gray, axis=1)
        is_gutter = row_std < 6.0

        segments: list[tuple[int, int]] = []
        in_panel = False
        start_y = 0
        gutter_run = 0

        for y in range(h):
            if not is_gutter[y]:
                if not in_panel:
                    start_y = y
                    in_panel = True
                gutter_run = 0
            else:
                if in_panel:
                    gutter_run += 1
                    if gutter_run >= self.min_gutter_px:
                        end_y = y - gutter_run
                        if end_y - start_y >= self.min_panel_height:
                            segments.append((start_y, end_y))
                        in_panel = False
                        gutter_run = 0

        if in_panel and h - start_y >= self.min_panel_height:
            segments.append((start_y, h))

        if not segments:
            segments = [(0, h)]

        panels_dir.mkdir(parents=True, exist_ok=True)
        panels: list[PanelInfo] = []
        for idx, (y1, y2) in enumerate(segments, start=1):
            panel_id = f"{page_id}_P{idx}"
            crop = img[y1:y2, 0:w]
            crop_path = panels_dir / f"{panel_id}.jpg"
            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            bbox = BBox(x1=0, y1=y1, x2=w, y2=y2)
            panels.append(
                PanelInfo(
                    panel_id=panel_id,
                    chapter_id=chapter_id,
                    page_id=page_id,
                    index_in_page=idx,
                    global_index=idx,
                    bbox=bbox,
                    width=w,
                    height=y2 - y1,
                    crop_path=str(crop_path.resolve()),
                )
            )

        return PageInfo(
            page_id=page_id,
            chapter_id=chapter_id,
            page_index=page_index,
            source_path=str(image_path.resolve()),
            format="manhwa",
            width=w,
            height=h,
            panels=panels,
        )
