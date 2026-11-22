"""ONNX YOLO Manga panel, face, body, and text detector with RTL/LTR sorting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from src.cv.registry import DETECTOR_REGISTRY
from src.schemas.ingestion import BBox, FocalPoint, PageInfo, PanelInfo


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thresh]
    return keep


def trim_white_border_with_offset(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop dominant empty white margins on inset pages and return (trimmed_img, x_offset, y_offset)."""
    ih, iw = img.shape[:2]
    if ih < 200 or iw < 200:
        return img, 0, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if float(np.mean(gray > 245)) <= 0.75:
        return img, 0, 0
    ys, xs = np.where(gray < 230)
    if len(xs) < 100:
        return img, 0, 0
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    area_ratio = ((x1 - x0) * (y1 - y0)) / float(ih * iw)
    if 0.05 <= area_ratio <= 0.60 and (x1 - x0) >= 120 and (y1 - y0) >= 120:
        pad_x = int(iw * 0.06)
        pad_y = int(ih * 0.06)
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(iw, x1 + pad_x)
        y1 = min(ih, y1 + pad_y)
        return img[y0:y1, x0:x1], x0, y0
    return img, 0, 0


def _row_cluster_sort(boxes: list[BBox], reverse_x: bool) -> list[BBox]:
    """Fallback row-clustering sort when panels overlap diagonally."""
    by_y = sorted(boxes, key=lambda b: (b.y1 + b.y2) / 2.0)
    rows: list[list[BBox]] = []
    for box in by_y:
        cy = (box.y1 + box.y2) / 2.0
        placed = False
        for row in rows:
            row_y1 = min(r.y1 for r in row)
            row_y2 = max(r.y2 for r in row)
            overlap = min(box.y2, row_y2) - max(box.y1, row_y1)
            min_h = max(1, min(box.height, row_y2 - row_y1))
            if overlap / min_h > 0.35 or (row_y1 <= cy <= row_y2):
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    rows.sort(key=lambda r: min(b.y1 for b in r))
    ordered: list[BBox] = []
    for row in rows:
        row.sort(key=lambda b: (b.x1 + b.x2) / 2.0, reverse=reverse_x)
        ordered.extend(row)
    return ordered


def _xy_cut_sort(boxes: list[BBox], reverse_x: bool) -> list[BBox]:
    """Recursively partition 2D comic layouts (including L-shape splits) via XY-Cut."""
    if len(boxes) <= 1:
        return list(boxes)

    # 1) Try horizontal cut (separating top band from bottom band cleanly)
    by_y = sorted(boxes, key=lambda b: ((b.y1 + b.y2) / 2.0, b.y1))
    best_y_gap = float("-inf")
    best_y_split = -1
    for i in range(1, len(by_y)):
        top_grp = by_y[:i]
        bot_grp = by_y[i:]
        top_bottom = max(b.y2 for b in top_grp)
        bot_top = min(b.y1 for b in bot_grp)
        max_top_cy = max((b.y1 + b.y2) / 2.0 for b in top_grp)
        min_bot_cy = min((b.y1 + b.y2) / 2.0 for b in bot_grp)
        if max_top_cy >= min_bot_cy:
            continue
        min_h = max(1, min(b.height for b in boxes))
        overlap = top_bottom - bot_top
        if overlap <= 0.22 * min_h:
            gap = bot_top - top_bottom
            if gap > best_y_gap:
                best_y_gap = gap
                best_y_split = i

    if best_y_split > 0:
        return _xy_cut_sort(by_y[:best_y_split], reverse_x) + _xy_cut_sort(
            by_y[best_y_split:], reverse_x
        )

    # 2) Try vertical cut (handles tall vertical panel alongside stacked horizontal panels)
    by_x = sorted(boxes, key=lambda b: ((b.x1 + b.x2) / 2.0, b.x1))
    best_x_gap = float("-inf")
    best_x_split = -1
    for i in range(1, len(by_x)):
        left_grp = by_x[:i]
        right_grp = by_x[i:]
        left_right = max(b.x2 for b in left_grp)
        right_left = min(b.x1 for b in right_grp)
        max_left_cx = max((b.x1 + b.x2) / 2.0 for b in left_grp)
        min_right_cx = min((b.x1 + b.x2) / 2.0 for b in right_grp)
        if max_left_cx >= min_right_cx:
            continue
        min_w = max(1, min(b.width for b in boxes))
        overlap = left_right - right_left
        if overlap <= 0.22 * min_w:
            gap = right_left - left_right
            if gap > best_x_gap:
                best_x_gap = gap
                best_x_split = i

    if best_x_split > 0:
        left_sorted = _xy_cut_sort(by_x[:best_x_split], reverse_x)
        right_sorted = _xy_cut_sort(by_x[best_x_split:], reverse_x)
        return (right_sorted + left_sorted) if reverse_x else (left_sorted + right_sorted)

    return _row_cluster_sort(boxes, reverse_x)


def sort_panels_reading_order(boxes: list[BBox], reading_order: str = "rtl") -> list[BBox]:
    """Sort 2D manga/comic panel boxes into reading order using Recursive XY-Cut."""
    if len(boxes) <= 1:
        return list(boxes)
    reverse_x = reading_order.lower() == "rtl"
    return _xy_cut_sort(boxes, reverse_x=reverse_x)


@DETECTOR_REGISTRY.register("manga_yolo")
class MangaYoloDetector:
    """Detects manga frames, faces, bodies, and text blocks via Manga109 ONNX YOLO."""

    name = "manga_yolo"

    def __init__(
        self,
        model_path: Path | str = "models/manga109_yolo/v2023.12.07_m_yv11/model.onnx",
        labels_path: Path | str = "models/manga109_yolo/v2023.12.07_m_yv11/labels.json",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self._session: ort.InferenceSession | None = None
        self.labels = ["body", "face", "frame", "text"]
        if self.labels_path.exists():
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))

    def _get_session(self) -> ort.InferenceSession | None:
        if self._session is None and self.model_path.exists():
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        return self._session

    def _run_yolo(self, img: np.ndarray) -> dict[str, list[tuple[BBox, float]]]:
        results: dict[str, list[tuple[BBox, float]]] = {lbl: [] for lbl in self.labels}
        sess = self._get_session()
        if sess is None:
            return results

        h, w = img.shape[:2]
        input_meta = sess.get_inputs()[0]
        target_size = 640
        if isinstance(input_meta.shape[2], int):
            target_size = input_meta.shape[2]

        scale = min(target_size / w, target_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        preds = sess.run(None, {input_meta.name: blob})[0]
        # Handle (1, 4+C, N) or (1, N, 4+C)
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))
        det = preds[0]

        if det.shape[1] == 6:
            # Post-NMS output: [x1, y1, x2, y2, score, cls]
            for row in det:
                score = float(row[4])
                cls_id = int(row[5])
                if score < self.conf_thresh or cls_id >= len(self.labels):
                    continue
                x1 = int(np.clip((row[0] - pad_x) / scale, 0, w))
                y1 = int(np.clip((row[1] - pad_y) / scale, 0, h))
                x2 = int(np.clip((row[2] - pad_x) / scale, 0, w))
                y2 = int(np.clip((row[3] - pad_y) / scale, 0, h))
                if x2 - x1 > 12 and y2 - y1 > 12:
                    results[self.labels[cls_id]].append((BBox(x1=x1, y1=y1, x2=x2, y2=y2), score))
            return results

        # Raw YOLO output: [cx, cy, bw, bh, cls0, cls1, ...]
        boxes_xywh = det[:, :4]
        cls_scores = det[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        max_scores = cls_scores[np.arange(len(det)), cls_ids]
        mask = max_scores >= self.conf_thresh

        boxes_xywh = boxes_xywh[mask]
        cls_ids = cls_ids[mask]
        max_scores = max_scores[mask]

        if len(boxes_xywh) == 0:
            return results

        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_x) / scale
        xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_y) / scale
        xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_x) / scale
        xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_y) / scale

        for cid, label in enumerate(self.labels):
            cmask = cls_ids == cid
            if not np.any(cmask):
                continue
            cboxes = xyxy[cmask]
            cscores = max_scores[cmask]
            keep = _nms(cboxes, cscores, self.iou_thresh)
            for k in keep:
                x1 = int(np.clip(cboxes[k, 0], 0, w))
                y1 = int(np.clip(cboxes[k, 1], 0, h))
                x2 = int(np.clip(cboxes[k, 2], 0, w))
                y2 = int(np.clip(cboxes[k, 3], 0, h))
                if x2 - x1 > 12 and y2 - y1 > 12:
                    results[label].append((BBox(x1=x1, y1=y1, x2=x2, y2=y2), float(cscores[k])))

        return results

    def detect_page(
        self,
        image_path: Path,
        chapter_id: str,
        page_index: int,
        panels_dir: Path,
        reading_order: str = "rtl",
    ) -> PageInfo:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read manga page: {image_path}")

        h, w = img.shape[:2]
        page_id = f"{chapter_id}_p{page_index:03d}"
        dets = self._run_yolo(img)

        frame_boxes = [b for b, _ in dets.get("frame", []) if b.area > (w * h * 0.015)]
        if not frame_boxes:
            frame_boxes = [BBox(x1=0, y1=0, x2=w, y2=h)]

        focal_candidates: list[tuple[str, BBox, float]] = []
        for kind in ("face", "body", "text"):
            for fb, fscore in dets.get(kind, []):
                focal_candidates.append((kind, fb, fscore))

        # Filter out extreme slivers / empty screentone strips using page-relative dimensions
        page_short = max(1, min(w, h))
        min_panel_side = max(32, int(page_short * 0.085))
        min_unfocused_side = max(48, int(page_short * 0.12))
        usable_frames: list[BBox] = []
        for fb in frame_boxes:
            pw_b = max(1, fb.width)
            ph_b = max(1, fb.height)
            ar_b = pw_b / ph_b
            has_face = False
            has_body = False
            has_text = False
            for kind, cand, _ in focal_candidates:
                cx = (cand.x1 + cand.x2) / 2.0
                cy = (cand.y1 + cand.y2) / 2.0
                if fb.x1 <= cx <= fb.x2 and fb.y1 <= cy <= fb.y2:
                    if kind == "face":
                        has_face = True
                    elif kind == "body":
                        has_body = True
                    elif kind == "text":
                        has_text = True
            crop_test = img[fb.y1 : fb.y2, fb.x1 : fb.x2]
            crop_std = float(crop_test.std()) if crop_test.size else 0.0

            if min(pw_b, ph_b) < min_panel_side:
                continue
            if min(pw_b, ph_b) < min_unfocused_side and not (has_face or has_text):
                continue
            if (ar_b < 0.19 or ar_b > 4.6) and not (has_face or has_text):
                continue
            if (ar_b < 0.30 or ar_b > 3.3) and not (has_face or has_body or has_text):
                continue
            if crop_std < 18.0 and not (has_face or has_text):
                continue
            usable_frames.append(fb)

        if usable_frames:
            frame_boxes = usable_frames

        ordered_frames = sort_panels_reading_order(frame_boxes, reading_order=reading_order)
        panels_dir.mkdir(parents=True, exist_ok=True)
        panels: list[PanelInfo] = []

        for idx, fb in enumerate(ordered_frames, start=1):
            panel_id = f"{page_id}_P{idx}"
            raw_crop = img[fb.y1 : fb.y2, fb.x1 : fb.x2]
            crop, trim_x0, trim_y0 = trim_white_border_with_offset(raw_crop)
            crop_path = panels_dir / f"{panel_id}.jpg"
            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            adj_fb = BBox(
                x1=fb.x1 + trim_x0,
                y1=fb.y1 + trim_y0,
                x2=fb.x1 + trim_x0 + crop.shape[1],
                y2=fb.y1 + trim_y0 + crop.shape[0],
            )
            pw = max(1, crop.shape[1])
            ph = max(1, crop.shape[0])
            focal_points: list[FocalPoint] = []
            for kind, cand, fscore in focal_candidates:
                cx = (cand.x1 + cand.x2) / 2.0
                cy = (cand.y1 + cand.y2) / 2.0
                if fb.x1 <= cx <= fb.x2 and fb.y1 <= cy <= fb.y2:
                    lx1 = max(0, cand.x1 - adj_fb.x1)
                    ly1 = max(0, cand.y1 - adj_fb.y1)
                    lx2 = min(pw, cand.x2 - adj_fb.x1)
                    ly2 = min(ph, cand.y2 - adj_fb.y1)
                    if lx2 <= lx1 or ly2 <= ly1:
                        continue
                    local_box = BBox(x1=lx1, y1=ly1, x2=lx2, y2=ly2)
                    shifted_cx = float(np.clip((cx - adj_fb.x1) / pw, 0.0, 1.0))
                    shifted_cy = float(np.clip((cy - adj_fb.y1) / ph, 0.0, 1.0))
                    focal_points.append(
                        FocalPoint(
                            kind=kind,
                            bbox=local_box,
                            norm_cx=shifted_cx,
                            norm_cy=shifted_cy,
                            norm_w=local_box.width / pw,
                            norm_h=local_box.height / ph,
                            confidence=fscore,
                        )
                    )

            panels.append(
                PanelInfo(
                    panel_id=panel_id,
                    chapter_id=chapter_id,
                    page_id=page_id,
                    index_in_page=idx,
                    global_index=idx,
                    bbox=adj_fb,
                    width=pw,
                    height=ph,
                    crop_path=str(crop_path.resolve()),
                    focal_points=focal_points,
                )
            )

        return PageInfo(
            page_id=page_id,
            chapter_id=chapter_id,
            page_index=page_index,
            source_path=str(image_path.resolve()),
            format="manga",
            width=w,
            height=h,
            panels=panels,
        )
