"""Schemas for Multi-Volume Series Continuity (Portrait Vault, 3-Tier Memory, Glossary & Handoff)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SeriesCharacter(BaseModel):
    """Persistent cross-volume character record backed by the Series Portrait Vault."""

    char_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = "Nhân vật"
    appearance: str = ""
    personality_notes: str = ""
    relationships: list[str] = Field(default_factory=list)
    status: Literal["active", "inactive", "deceased", "unknown"] = "active"
    first_seen_vol: int = 1
    last_seen_vol: int = 1
    first_revealed_vol: int | None = None
    portrait_path: str = ""
    portrait_vol: int = 1


class VolumeHandoff(BaseModel):
    """Hot Memory: exact ending state of Volume k-1 to bridge seamlessly into Volume k."""

    volume_index: int
    job_id: str
    ending_summary: str = ""
    cliffhanger_hook: str = ""
    last_beats_tail: list[str] = Field(default_factory=list)
    ending_mood: str = "tense"
    ending_characters: list[str] = Field(default_factory=list)
    last_bgm_track_id: str = ""


class VolumeLedgerEntry(BaseModel):
    """Cold Memory: compact 3-5 sentence digest & milestones for a completed volume."""

    volume_index: int
    job_id: str
    input_path: str = ""
    title_vi: str = ""
    summary_vi: str = ""
    key_milestones: list[str] = Field(default_factory=list)
    characters_featured: list[str] = Field(default_factory=list)
    resolved_threads: list[str] = Field(default_factory=list)
    duration_sec: float = 0.0
    output_mp4: str = ""


class SeriesGlossary(BaseModel):
    """Human-editable YAML glossary locking terminology, pronouns, and name overrides across all volumes."""

    series_title: str = ""
    terminology: dict[str, str] = Field(default_factory=dict)
    pronoun_pairs: dict[str, str] = Field(default_factory=dict)
    pronunciation_map: dict[str, str] = Field(default_factory=dict)
    character_name_overrides: dict[str, str] = Field(default_factory=dict)


class SeriesState(BaseModel):
    """Master persistent state stored at series/<series_id>/series_state.json."""

    series_id: str
    series_title: str = ""
    genre_guess: str = ""
    tone_summary: str = ""
    persona: str = "kich_tinh"
    tts_voice: str = ""
    music_genre_prefs: dict[str, float] = Field(default_factory=dict)
    recent_bgm_track_ids: list[str] = Field(default_factory=list)
    characters: list[SeriesCharacter] = Field(default_factory=list)
    active_plot_threads: list[str] = Field(default_factory=list)
    ledger: list[VolumeLedgerEntry] = Field(default_factory=list)
    last_handoff: VolumeHandoff | None = None


class SeriesContext(BaseModel):
    """Bounded runtime context injected into Stages 1.5-6 for a specific volume."""

    series_id: str
    series_dir: Path
    volume_index: int = 1
    total_volumes: int | None = None
    continuity_mode: Literal["standalone", "episodic", "seamless"] = "episodic"
    state: SeriesState
    glossary: SeriesGlossary = Field(default_factory=SeriesGlossary)
