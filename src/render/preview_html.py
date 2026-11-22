"""Standalone HTML storyboard preview generator."""

from __future__ import annotations

import html
from pathlib import Path

from src.schemas.timeline import VideoTimeline


def generate_preview_html(timeline: VideoTimeline, output_path: Path) -> Path:
    """Create a self-contained HTML storyboard inspecting every shot and subtitle."""
    rows: list[str] = []
    for shot in timeline.shots:
        dur = (shot.end_ms - shot.start_ms) / 1000.0
        panel_lbl = html.escape(shot.panel_id) + (
            f" + {html.escape(shot.secondary_panel_id)}" if shot.secondary_panel_id else ""
        )
        rows.append(
            f"<tr><td>{html.escape(shot.shot_id)}</td>"
            f"<td>{panel_lbl}</td>"
            f"<td>{shot.start_ms / 1000:.2f}s - {shot.end_ms / 1000:.2f}s ({dur:.2f}s)</td>"
            f"<td>{html.escape(shot.camera_move)} ({html.escape(shot.transition_in)})</td>"
            f"<td>{html.escape(shot.mood)}</td></tr>"
        )

    doc = f"""<!doctype html>
<html lang="vi">
<head><meta charset="utf-8"><title>Preview {html.escape(timeline.job_id)}</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;padding:24px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #333;padding:8px;text-align:left}}
th{{background:#222}}</style></head>
<body>
<h1>Storyboard: {html.escape(timeline.job_id)}</h1>
<p>Resolution: {timeline.width}x{timeline.height}@{timeline.fps}fps | Duration: {timeline.total_duration_ms/1000:.2f}s | Shots: {len(timeline.shots)}</p>
<table><thead><tr><th>Shot</th><th>Panel</th><th>Time</th><th>Camera</th><th>Mood</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
