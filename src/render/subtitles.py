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


@lru_cache(maxsize=512)
def _render_subtitle_patch(
    text: str,
    width: int,
    height: int,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    """Pre-render subtitle pill RGBA patch once and cache pre-multiplied BGR + inverse alpha."""
    font = _load_font(max(18, int(height * 0.036)))
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = dummy.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 22, 12
    box_w = min(max(32, width - 40), tw + 2 * pad_x)
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
    tx = (width - tw) // 2 - px0
    ty = ry0 + pad_y - bbox[1]
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

    rgba = np.asarray(patch, dtype=np.uint16)
    fg_bgr = rgba[:, :, [2, 1, 0]]
    alpha = rgba[:, :, 3:4]
    fg_premul = fg_bgr * alpha
    inv_alpha = 255 - alpha
    return px0, py0, px1, py1, fg_premul, inv_alpha


def warm_subtitle_cache(texts: list[str], width: int, height: int) -> None:
    """Pre-render unique subtitle patches on the main thread before parallel compositing."""
    seen: set[str] = set()
    for text in texts:
        if text and text.strip() and text not in seen:
            seen.add(text)
            _render_subtitle_patch(text, width, height)


def draw_subtitle_overlay(
    frame_bgr: np.ndarray,
    text: str,
    font_size: int = 28,
    in_place: bool = False,
) -> np.ndarray:
    """Render Vietnamese UTF-8 subtitle pill at the bottom center of frame_bgr using cached ROI blend."""
    if not text.strip():
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    px0, py0, px1, py1, fg_premul, inv_alpha = _render_subtitle_patch(text, w, h)
    out = frame_bgr if in_place else frame_bgr.copy()
    roi = out[py0:py1, px0:px1].astype(np.uint16)
    blended = (roi * inv_alpha + fg_premul + 127) // 255
    out[py0:py1, px0:px1] = blended.astype(np.uint8)
    return out
