"""CapCut PC draft_content.json exporter for human-in-the-loop editing."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from src.exporters.registry import EXPORTER_REGISTRY
from src.schemas.timeline import VideoTimeline


@EXPORTER_REGISTRY.register("capcut_draft")
class CapCutDraftExporter:
    """Exports an editable CapCut PC draft directory (draft_content.json + draft_meta_info.json)."""

    name = "capcut_draft"

    def export(self, timeline: VideoTimeline, job_dir: Path, **kwargs: Any) -> Path:
        draft_dir = job_dir / "capcut_draft"
        draft_dir.mkdir(parents=True, exist_ok=True)

        video_materials = []
        video_segments = []
        diptych_segments = []
        for shot in timeline.shots:
            mat_id = str(uuid.uuid4())
            seg_id = str(uuid.uuid4())
            start_us = shot.start_ms * 1000
            dur_us = max(1000, (shot.end_ms - shot.start_ms) * 1000)
            video_materials.append(
                {
                    "id": mat_id,
                    "type": "photo",
                    "path": shot.panel_path,
                    "material_name": shot.panel_id,
                    "width": timeline.width,
                    "height": timeline.height,
                }
            )
            video_segments.append(
                {
                    "id": seg_id,
                    "material_id": mat_id,
                    "target_timerange": {"start": start_us, "duration": dur_us},
                    "source_timerange": {"start": 0, "duration": dur_us},
                }
            )
            if shot.secondary_panel_path:
                sec_mat_id = str(uuid.uuid4())
                video_materials.append(
                    {
                        "id": sec_mat_id,
                        "type": "photo",
                        "path": shot.secondary_panel_path,
                        "material_name": shot.secondary_panel_id or f"{shot.panel_id}_2",
                        "width": timeline.width,
                        "height": timeline.height,
                    }
                )
                diptych_segments.append(
                    {
                        "id": str(uuid.uuid4()),
                        "material_id": sec_mat_id,
                        "target_timerange": {"start": start_us, "duration": dur_us},
                        "source_timerange": {"start": 0, "duration": dur_us},
                    }
                )

        audio_id = str(uuid.uuid4())
        audio_materials = [
            {
                "id": audio_id,
                "type": "extract_music",
                "path": timeline.voiceover_path,
                "name": "voiceover",
                "duration": timeline.total_duration_ms * 1000,
            }
        ]
        audio_segments = [
            {
                "id": str(uuid.uuid4()),
                "material_id": audio_id,
                "target_timerange": {"start": 0, "duration": timeline.total_duration_ms * 1000},
                "source_timerange": {"start": 0, "duration": timeline.total_duration_ms * 1000},
            }
        ]
        if timeline.bgm_path:
            bgm_id = str(uuid.uuid4())
            audio_materials.append(
                {
                    "id": bgm_id,
                    "type": "music",
                    "path": timeline.bgm_path,
                    "name": "bgm",
                    "duration": timeline.total_duration_ms * 1000,
                }
            )
            audio_segments.append(
                {
                    "id": str(uuid.uuid4()),
                    "material_id": bgm_id,
                    "target_timerange": {"start": 0, "duration": timeline.total_duration_ms * 1000},
                    "source_timerange": {"start": 0, "duration": timeline.total_duration_ms * 1000},
                }
            )

        text_materials = []
        text_segments = []
        for sub in timeline.subtitles:
            tid = str(uuid.uuid4())
            s_us = sub.start_ms * 1000
            d_us = max(1000, (sub.end_ms - sub.start_ms) * 1000)
            text_materials.append(
                {
                    "id": tid,
                    "type": "subtitle",
                    "content": sub.text,
                }
            )
            text_segments.append(
                {
                    "id": str(uuid.uuid4()),
                    "material_id": tid,
                    "target_timerange": {"start": s_us, "duration": d_us},
                    "source_timerange": {"start": 0, "duration": d_us},
                }
            )

        tracks = [
            {"id": str(uuid.uuid4()), "type": "video", "segments": video_segments},
            {"id": str(uuid.uuid4()), "type": "audio", "segments": audio_segments},
        ]
        if diptych_segments:
            tracks.append({"id": str(uuid.uuid4()), "type": "video", "segments": diptych_segments})
        if text_segments:
            tracks.append({"id": str(uuid.uuid4()), "type": "text", "segments": text_segments})

        draft_content = {
            "id": str(uuid.uuid4()),
            "name": timeline.job_id,
            "duration": timeline.total_duration_ms * 1000,
            "canvas_config": {"width": timeline.width, "height": timeline.height, "ratio": "custom"},
            "materials": {
                "videos": video_materials,
                "audios": audio_materials,
                "texts": text_materials,
            },
            "tracks": tracks,
        }
        (draft_dir / "draft_content.json").write_text(
            json.dumps(draft_content, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (draft_dir / "draft_meta_info.json").write_text(
            json.dumps({"draft_name": timeline.job_id, "draft_id": draft_content["id"]}, indent=2),
            encoding="utf-8",
        )
        return draft_dir
