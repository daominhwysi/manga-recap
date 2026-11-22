"""Rendering engine: frame compositor, subtitles, and FFmpeg video output."""

from src.render.compositor import render_shot_frame
from src.render.ffmpeg_renderer import FFmpegOpenCVRenderer
from src.render.preview_html import generate_preview_html
from src.render.registry import RENDERER_REGISTRY
from src.render.subtitles import draw_subtitle_overlay, write_srt

__all__ = [
    "FFmpegOpenCVRenderer",
    "RENDERER_REGISTRY",
    "draw_subtitle_overlay",
    "generate_preview_html",
    "render_shot_frame",
    "write_srt",
]
