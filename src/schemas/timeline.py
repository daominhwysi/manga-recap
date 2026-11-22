"""Schemas for Stage 4 & 5: Audio synthesis, subtitles, and video timeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudioBeatInfo(BaseModel):
    beat_id: str
    audio_path: str
    duration_ms: int
    start_ms: int = 0
    end_ms: int = 0
    pause_after_ms: int = 240
    sample_rate: int = 48000


class AudioManifest(BaseModel):
    backend: str
    voice: str
    sample_rate: int = 48000
    total_duration_ms: int
    voiceover_path: str
    beats: list[AudioBeatInfo] = Field(default_factory=list)


class TimelineShot(BaseModel):
    shot_id: str
    beat_id: str
    panel_id: str
    panel_path: str
    secondary_panel_id: str | None = None
    secondary_panel_path: str | None = None
    start_ms: int
    end_ms: int
    camera_move: str = "zoom_in"  # zoom_in, zoom_out, focus_face, shake_impact, pan_down, pan_up, pan_left, pan_right, diptych
    focus_x: float = 0.5
    focus_y: float = 0.5
    zoom_start: float = 1.0
    zoom_end: float = 1.15
    transition_in: str = "cut"  # cut, crossfade, flash_white, fade_black
    mood: str = "tense"


class SubtitleCue(BaseModel):
    index: int
    start_ms: int
    end_ms: int
    text: str


class VideoTimeline(BaseModel):
    job_id: str
    width: int
    height: int
    fps: int
    total_duration_ms: int
    voiceover_path: str
    bgm_path: str | None = None
    shots: list[TimelineShot] = Field(default_factory=list)
    subtitles: list[SubtitleCue] = Field(default_factory=list)
