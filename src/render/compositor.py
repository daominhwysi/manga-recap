"""OpenCV frame compositor with blurred manga background, drop shadow, diptychs, and Ken Burns motion."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.schemas.timeline import TimelineShot


def _make_blurred_bg(panel_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Build dark blurred background at 1/4 resolution then upscale for fast silky bokeh."""
    sw = max(32, width // 4)
    sh = max(18, height // 4)
    ph, pw = panel_bgr.shape[:2]
    scale = max(sw / max(1, pw), sh / max(1, ph))
    nw, nh = max(sw, int(pw * scale)), max(sh, int(ph * scale))
    resized = cv2.resize(panel_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    x0 = (nw - sw) // 2
    y0 = (nh - sh) // 2
    crop = resized[y0 : y0 + sh, x0 : x0 + sw]
    blurred = cv2.GaussianBlur(crop, (15, 15), 0)
    darkened = ((blurred.astype(np.uint16) * 82) >> 8).astype(np.uint8)
    return cv2.resize(darkened, (width, height), interpolation=cv2.INTER_LINEAR)


def _render_single_panel(
    panel_bgr: np.ndarray,
    shot: TimelineShot,
    ease: float,
    t: float,
    max_w: int,
    max_h: int,
) -> np.ndarray:
    ph, pw = panel_bgr.shape[:2]
    zoom = max(1.0, shot.zoom_start + (shot.zoom_end - shot.zoom_start) * ease)

    fit_scale = min(max_w / max(1, pw), max_h / max(1, ph))
    disp_w = max(2, int(round(pw * fit_scale)))
    disp_h = max(2, int(round(ph * fit_scale)))

    # Aspect-aware focal travel for pan_* moves
    fx_norm = shot.focus_x
    fy_norm = shot.focus_y
    if shot.camera_move == "pan_down":
        fy_norm = 0.22 + 0.56 * ease
    elif shot.camera_move == "pan_up":
        fy_norm = 0.78 - 0.56 * ease
    elif shot.camera_move == "pan_left":
        fx_norm = 0.78 - 0.56 * ease
    elif shot.camera_move == "pan_right":
        fx_norm = 0.22 + 0.56 * ease

    win_w = pw / zoom
    win_h = ph / zoom
    fx_px = float(np.clip(fx_norm * pw, win_w / 2.0, max(win_w / 2.0, pw - win_w / 2.0)))
    fy_px = float(np.clip(fy_norm * ph, win_h / 2.0, max(win_h / 2.0, ph - win_h / 2.0)))

    total_scale_x = disp_w / max(1e-3, win_w)
    total_scale_y = disp_h / max(1e-3, win_h)
    tx = (disp_w / 2.0) - fx_px * total_scale_x
    ty = (disp_h / 2.0) - fy_px * total_scale_y

    if total_scale_x > 1.8 or total_scale_y > 1.8:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR

    mat = np.array([[total_scale_x, 0.0, tx], [0.0, total_scale_y, ty]], dtype=np.float32)
    return cv2.warpAffine(
        panel_bgr,
        mat,
        (disp_w, disp_h),
        flags=interp,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _blit_with_shadow(
    canvas: np.ndarray,
    fg: np.ndarray,
    px: int,
    py: int,
    border: int = 3,
    shadow_offset: int = 7,
) -> None:
    ch, cw = canvas.shape[:2]
    fh, fw = fg.shape[:2]

    bx1 = max(0, px - border)
    by1 = max(0, py - border)
    bx2 = min(cw, px + fw + border)
    by2 = min(ch, py + fh + border)

    # Darken ONLY the exposed shadow strips (right + bottom) instead of the whole occluded panel area
    sx1 = max(0, bx1 + shadow_offset)
    sy1 = max(0, by1 + shadow_offset)
    sx2 = min(cw, bx2 + shadow_offset)
    sy2 = min(ch, by2 + shadow_offset)
    if sx2 > bx2 and sy2 > sy1:
        r_roi = canvas[sy1:sy2, bx2:sx2]
        canvas[sy1:sy2, bx2:sx2] = ((r_roi.astype(np.uint16) * 107) >> 8).astype(np.uint8)
    if sy2 > by2 and bx2 > sx1:
        b_roi = canvas[by2:sy2, sx1:bx2]
        canvas[by2:sy2, sx1:bx2] = ((b_roi.astype(np.uint16) * 107) >> 8).astype(np.uint8)

    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (242, 242, 242), -1)
    canvas[py : py + fh, px : px + fw] = fg


def render_shot_frame(
    panel_bgr: np.ndarray,
    shot: TimelineShot,
    progress: float,
    width: int,
    height: int,
    bg_cache: dict[str, np.ndarray] | None = None,
    secondary_bgr: np.ndarray | None = None,
    margin_ratio: float = 0.06,
    bottom_safe_ratio: float = 0.14,
    out_buf: np.ndarray | None = None,
    reading_order: str = "rtl",
) -> np.ndarray:
    """Compose a single 1080p/720p frame (single or diptych) above subtitle safe area."""
    t = float(np.clip(progress, 0.0, 1.0))
    ease = t * t * (3.0 - 2.0 * t)

    if bg_cache is not None and shot.panel_id in bg_cache:
        bg = bg_cache[shot.panel_id]
    else:
        bg = _make_blurred_bg(panel_bgr, width, height)
        if bg_cache is not None:
            bg_cache[shot.panel_id] = bg

    if out_buf is not None and out_buf.shape == bg.shape:
        np.copyto(out_buf, bg)
        canvas = out_buf
    else:
        canvas = bg.copy()

    margin_x = int(width * margin_ratio)
    margin_top = int(height * margin_ratio)
    margin_bottom = int(height * max(margin_ratio, bottom_safe_ratio))
    max_w = max(64, width - 2 * margin_x)
    max_h = max(64, height - margin_top - margin_bottom)

    if shot.camera_move == "diptych" and secondary_bgr is not None:
        # Render two vertical panels side-by-side (RTL manga flow: first panel right, second panel left)
        gap = max(12, int(width * 0.015))
        zoom = 1.0 + 0.06 * ease
        h1, w1 = panel_bgr.shape[:2]
        h2, w2 = secondary_bgr.shape[:2]
        norm_w1 = w1 / max(1, h1)
        norm_w2 = w2 / max(1, h2)
        target_h = min(max_h, int((max_w - gap) / max(0.2, norm_w1 + norm_w2)))
        target_h = max(48, int(target_h * zoom))
        target_h = min(max_h, target_h)

        dw1 = max(24, int(round(target_h * norm_w1)))
        dw2 = max(24, int(round(target_h * norm_w2)))
        if dw1 + dw2 + gap > max_w:
            shrink = (max_w - gap) / max(1, dw1 + dw2)
            dw1 = max(24, int(dw1 * shrink))
            dw2 = max(24, int(dw2 * shrink))
            target_h = max(48, int(target_h * shrink))

        fg1 = cv2.resize(panel_bgr, (dw1, target_h), interpolation=cv2.INTER_LINEAR)
        fg2 = cv2.resize(secondary_bgr, (dw2, target_h), interpolation=cv2.INTER_LINEAR)

        total_w = dw1 + dw2 + gap
        start_x = int(np.clip((width - total_w) // 2, 6, max(6, width - total_w - 6)))
        py = int(np.clip(margin_top + (max_h - target_h) // 2, 6, max(6, height - target_h - 6)))
        if reading_order.lower() == "ltr":
            # LTR order: primary panel on Left, secondary panel on Right
            _blit_with_shadow(canvas, fg1, start_x, py)
            _blit_with_shadow(canvas, fg2, start_x + dw1 + gap, py)
        else:
            # RTL order: newer secondary panel on Left, primary panel on Right
            _blit_with_shadow(canvas, fg2, start_x, py)
            _blit_with_shadow(canvas, fg1, start_x + dw2 + gap, py)
        return canvas

    fg = _render_single_panel(panel_bgr, shot, ease, t, max_w=max_w, max_h=max_h)
    disp_h, disp_w = fg.shape[:2]

    dx = 0
    dy = 0
    if shot.camera_move == "shake_impact" and t < 0.45:
        amp = max(2, int(width * 0.006 * (1.0 - t / 0.45)))
        dx = int(round(math.sin(t * 45.0) * amp))
        dy = int(round(math.cos(t * 38.0) * amp * 0.7))

    px = int(np.clip((width - disp_w) // 2 + dx, 4, max(4, width - disp_w - 4)))
    py = int(np.clip(margin_top + (max_h - disp_h) // 2 + dy, 4, max(4, height - disp_h - 4)))
    _blit_with_shadow(canvas, fg, px, py)
    return canvas
