"""Standalone HTML storyboard preview generator with panel thumbnails, subtitles & audio player."""

from __future__ import annotations

import html
import os
from pathlib import Path

from src.schemas.timeline import VideoTimeline


def _rel_or_file(path_str: str, base_dir: Path) -> str:
    p = Path(path_str)
    try:
        return os.path.relpath(p.resolve(), base_dir.resolve()).replace("\\", "/")
    except Exception:
        return str(p)


def generate_preview_html(timeline: VideoTimeline, output_path: Path) -> Path:
    """Create a self-contained HTML storyboard inspecting panel thumbnails, audio, subtitles, and pacing."""
    base_dir = output_path.parent
    vo_src = _rel_or_file(timeline.voiceover_path, base_dir)
    audio_html = (
        f'<div class="audio-bar"><strong>Voiceover:</strong> '
        f'<audio controls preload="metadata" src="{html.escape(vo_src)}"></audio></div>'
        if timeline.voiceover_path
        else ""
    )

    rows: list[str] = []
    for shot in timeline.shots:
        dur = (shot.end_ms - shot.start_ms) / 1000.0
        img1 = html.escape(_rel_or_file(shot.panel_path, base_dir))
        thumbs = f'<img class="thumb" src="{img1}" alt="{html.escape(shot.panel_id)}" loading="lazy" />'
        if shot.secondary_panel_path:
            img2 = html.escape(_rel_or_file(shot.secondary_panel_path, base_dir))
            thumbs = (
                f'<img class="thumb" src="{img2}" alt="{html.escape(shot.secondary_panel_id or "")}" loading="lazy" /> '
                + thumbs
            )

        panel_lbl = html.escape(shot.panel_id) + (
            f" + {html.escape(shot.secondary_panel_id)}" if shot.secondary_panel_id else ""
        )
        matched_subs = [
            sub.text
            for sub in timeline.subtitles
            if sub.end_ms > shot.start_ms and sub.start_ms < shot.end_ms
        ]
        sub_txt = html.escape(" ".join(matched_subs)) if matched_subs else "<em>(Không lời)</em>"

        rows.append(
            f"<tr>"
            f"<td><strong>{html.escape(shot.shot_id)}</strong><br><small>{panel_lbl}</small></td>"
            f'<td class="thumbs">{thumbs}</td>'
            f"<td class=\"sub-col\">{sub_txt}</td>"
            f"<td>{shot.start_ms / 1000:.2f}s – {shot.end_ms / 1000:.2f}s<br><small>({dur:.2f}s)</small></td>"
            f'<td><span class="badge">{html.escape(shot.camera_move)}</span><br><small>{html.escape(shot.transition_in)}</small></td>'
            f'<td><span class="badge mood-{html.escape(shot.mood)}">{html.escape(shot.mood)}</span></td>'
            f"</tr>"
        )

    doc = f"""<!doctype html>
<html lang="vi">
<head><meta charset="utf-8"><title>Storyboard: {html.escape(timeline.job_id)}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0e1015;color:#e8eaf0;padding:24px;margin:0}}
h1{{margin:0 0 8px 0}}
.meta{{color:#9aa4b2;margin-bottom:14px}}
.audio-bar{{background:#171b24;padding:12px 16px;border-radius:8px;margin-bottom:18px;display:flex;align-items:center;gap:14px}}
audio{{width:420px;max-width:100%}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #262c3a;padding:8px 10px;text-align:left;vertical-align:middle}}
th{{background:#181d27;position:sticky;top:0}}
.thumbs{{white-space:nowrap}}
.thumb{{height:110px;max-width:180px;object-fit:contain;background:#07080b;border-radius:4px;border:1px solid #303748;margin-right:4px}}
.sub-col{{max-width:340px;line-height:1.45}}
.badge{{display:inline-block;padding:2px 8px;border-radius:99px;background:#273043;font-size:12px}}
.mood-action{{background:#5e2228;color:#ffd0d4}}
.mood-tense{{background:#4d341a;color:#ffe2b8}}
.mood-tragic{{background:#2e294e;color:#d6d0ff}}
.mood-comedic{{background:#1f4a32;color:#c2f5d6}}
</style></head>
<body>
<h1>Storyboard: {html.escape(timeline.job_id)}</h1>
<p class="meta">Resolution: {timeline.width}x{timeline.height}@{timeline.fps}fps ({timeline.reading_order.upper()}) | Duration: {timeline.total_duration_ms/1000:.2f}s | Shots: {len(timeline.shots)} | Subtitles: {len(timeline.subtitles)}</p>
{audio_html}
<table><thead><tr><th>Shot</th><th>Panel(s)</th><th>Voiceover / Subtitle</th><th>Time</th><th>Camera</th><th>Mood</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
