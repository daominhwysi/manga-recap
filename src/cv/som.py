"""Set-of-Mark (SoM) annotator for VLM page comprehension."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.schemas.ingestion import PanelInfo


def render_som_page(
    page_image_path: Path,
    panels: list[PanelInfo],
    output_path: Path,
) -> Path:
    """Draw numbered [P1], [P2]... bounding boxes onto a copy of the manga page."""
    img = cv2.imread(str(page_image_path))
    if img is None:
        raise ValueError(f"Unable to read image: {page_image_path}")

    annotated = img.copy()
    palette = [
        (30, 50, 235),
        (220, 90, 20),
        (20, 165, 50),
        (190, 30, 175),
        (10, 145, 225),
    ]

    for idx, panel in enumerate(panels):
        color = palette[idx % len(palette)]
        b = panel.bbox
        cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), color, 3)

        label = f"P{panel.index_in_page}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        lx1 = max(0, b.x1)
        ly1 = max(0, b.y1)
        lx2 = min(annotated.shape[1], lx1 + tw + 14)
        ly2 = min(annotated.shape[0], ly1 + th + 12)
        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(
            annotated,
            label,
            (lx1 + 7, ly2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return output_path
