"""Stage 1 orchestrator: ingestion, format detection, panel extraction, and SoM."""

from __future__ import annotations

from pathlib import Path

import cv2

from src.config import CVConfig
from src.cv.loader import collect_chapters
from src.cv.manga_detector import MangaYoloDetector
from src.cv.manhwa_slicer import ManhwaStripSlicer
from src.cv.som import render_som_page
from src.schemas.ingestion import ChapterInfo, JobManifest


def run_stage1_ingest(
    job_id: str,
    input_path: Path,
    job_dir: Path,
    cv_config: CVConfig,
    force: bool = False,
) -> JobManifest:
    """Run Stage 1 CV ingestion with checkpointing."""
    manifest_path = job_dir / "01_manifest.json"
    if manifest_path.exists() and not force:
        return JobManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    chapters_raw = collect_chapters(
        input_path=input_path,
        work_dir=job_dir,
        max_pages_per_chapter=cv_config.max_pages_per_chapter,
    )

    manga_detector = MangaYoloDetector(
        model_path=cv_config.yolo_model_path,
        labels_path=cv_config.yolo_labels_path,
        conf_thresh=cv_config.confidence_threshold,
        iou_thresh=cv_config.iou_threshold,
    )
    manhwa_slicer = ManhwaStripSlicer(yolo_detector=manga_detector)

    panels_dir = job_dir / "panels"
    som_dir = job_dir / "som"
    chapters: list[ChapterInfo] = []
    global_idx = 1

    for cid, title, img_paths in chapters_raw:
        pages = []
        for p_idx, img_path in enumerate(img_paths, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            mode = cv_config.detector
            if mode == "auto":
                is_manhwa = (h / max(1, w)) >= 2.2
                detector = manhwa_slicer if is_manhwa else manga_detector
            elif mode == "manhwa_slicer":
                detector = manhwa_slicer
            else:
                detector = manga_detector

            page_info = detector.detect_page(
                image_path=img_path,
                chapter_id=cid,
                page_index=p_idx,
                panels_dir=panels_dir,
                reading_order=cv_config.reading_order,
            )

            for panel in page_info.panels:
                panel.global_index = global_idx
                global_idx += 1

            som_out = som_dir / f"{page_info.page_id}_som.jpg"
            render_som_page(img_path, page_info.panels, som_out)
            page_info.som_paths = [str(som_out.resolve())]
            pages.append(page_info)

        chapters.append(ChapterInfo(chapter_id=cid, title=title, pages=pages))

    manifest = JobManifest(
        job_id=job_id,
        input_path=str(input_path.resolve()),
        format_mode=cv_config.detector,
        reading_order=cv_config.reading_order,
        chapters=chapters,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
