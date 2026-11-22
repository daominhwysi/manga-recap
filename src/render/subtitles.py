"""Subtitle generation (SRT) and UTF-8 Pillow subtitle overlay."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.schemas.timeline import SubtitleCue


def _format_srt_time(ms: int) -> str:
    ms = max(0, ms)
    hrs = ms // 3_600_000
    mins = (ms % 3_600_000) // 60_000
    secs = (ms % 60_000) // 1000
    rem = ms % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{rem:03d}"


def write_srt(subtitles: list[SubtitleCue], output_path: Path) -> Path:
    """Write SubtitleCue list to standard SRT file."""
    lines: list[str] = []
    for cue in subtitles:
        lines.append(str(cue.index))
        lines.append(f"{_format_srt_time(cue.start_ms)} --> {_format_srt_time(cue.end_ms)}")
        lines.append(cue.text)
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def _wrap_subtitle_lines(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_text_w: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    """Word-wrap subtitle text across 1-2 balanced lines so narrow 9:16 canvases never clip."""
    words = text.strip().split()
    if not words:
        return []
    full_bbox = draw.textbbox((0, 0), text, font=font)
    if (full_bbox[2] - full_bbox[0]) <= max_text_w:
        return [text.strip()]

    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        tb = draw.textbbox((0, 0), trial, font=font)
        if cur and (tb[2] - tb[0]) > max_text_w:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


@lru_cache(maxsize=512)
def _render_subtitle_patch(
    text: str,
    width: int,
    height: int,
    font_ratio: float = 0.036,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    """Pre-render wrapped subtitle pill RGBA patch once and cache pre-multiplied BGR + inverse alpha."""
    base_dim = min(width, height) if height > width else height
    font_px = max(18, int(round(base_dim * font_ratio * (1.28 if height > width else 1.0))))
    font = _load_font(font_px)
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    pad_x, pad_y = 22, 12
    max_text_w = max(80, width - 48 - 2 * pad_x)
    lines = _wrap_subtitle_lines(text, font, max_text_w, dummy) or [text]

    line_metrics: list[tuple[str, int, int, int]] = []
    for ln in lines:
        bb = dummy.textbbox((0, 0), ln, font=font)
        line_metrics.append((ln, bb[2] - bb[0], bb[3] - bb[1], bb[1]))

    line_gap = max(4, int(font_px * 0.25))
    tw = max(m[1] for m in line_metrics)
    th = sum(m[2] for m in line_metrics) + line_gap * (len(line_metrics) - 1)
    box_w = min(max(32, width - 32), tw + 2 * pad_x)
    box_h = th + 2 * pad_y
    x0 = max(0, (width - box_w) // 2)
    y0 = max(0, height - box_h - int(height * 0.045))

    border_pad = 3
    px0 = max(0, x0 - border_pad)
    py0 = max(0, y0 - border_pad)
    px1 = min(width, x0 + box_w + border_pad)
    py1 = min(height, y0 + box_h + border_pad)
    pw = max(1, px1 - px0)
    ph = max(1, py1 - py0)

    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    rx0 = x0 - px0
    ry0 = y0 - py0
    draw.rounded_rectangle(
        [(rx0, ry0), (rx0 + box_w, ry0 + box_h)],
        radius=12,
        fill=(15, 15, 20, 215),
        outline=(255, 255, 255, 70),
        width=2,
    )
    cur_y = ry0 + pad_y
    for ln, lw, lh, y_off in line_metrics:
        tx = (width - lw) // 2 - px0
        draw.text((tx, cur_y - y_off), ln, font=font, fill=(255, 255, 255, 255))
        cur_y += lh + line_gap

    rgba = np.asarray(patch, dtype=np.uint16)
    fg_bgr = rgba[:, :, [2, 1, 0]]
    alpha = rgba[:, :, 3:4]
    fg_premul = fg_bgr * alpha
    inv_alpha = 255 - alpha
    return px0, py0, px1, py1, fg_premul, inv_alpha


def warm_subtitle_cache(
    texts: list[str],
    width: int,
    height: int,
    font_ratio: float = 0.036,
) -> None:
    """Pre-render unique subtitle patches on the main thread before parallel compositing."""
    seen: set[str] = set()
    for text in texts:
        if text and text.strip() and text not in seen:
            seen.add(text)
            _render_subtitle_patch(text, width, height, font_ratio=font_ratio)


def draw_subtitle_overlay(
    frame_bgr: np.ndarray,
    text: str,
    font_size: int = 28,
    in_place: bool = False,
    font_ratio: float = 0.036,
) -> np.ndarray:
    """Render wrapped Vietnamese UTF-8 subtitle pill at the bottom center of frame_bgr."""
    if not text.strip():
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    px0, py0, px1, py1, fg_premul, inv_alpha = _render_subtitle_patch(
        text, w, h, font_ratio=font_ratio
    )
    out = frame_bgr if in_place else frame_bgr.copy()
    roi = out[py0:py1, px0:px1].astype(np.uint16)
    blended = (roi * inv_alpha + fg_premul + 127) // 255
    out[py0:py1, px0:px1] = blended.astype(np.uint8)
    return out
