"""Schemas for Stage 3: Narrative Scriptwriter and Critic review."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScriptBeat(BaseModel):
    beat_id: str
    section: str = "setup"  # hook, setup, rising, climax, cliffhanger
    narration_vi: str
    panel_ids: list[str] = Field(default_factory=list)
    mood: str = "tense"  # tense, action, epic, calm, dramatic
    pause_after_ms: int = 240


class RecapScript(BaseModel):
    title_vi: str
    persona: str = "kich_tinh"
    hook_summary: str = ""
    beats: list[ScriptBeat] = Field(default_factory=list)


class CriticIssue(BaseModel):
    beat_id: str
    severity: str = "warning"  # warning, error
    category: str = "style"
    message: str


class CriticReport(BaseModel):
    passed: bool = True
    score: float = 10.0
    issues: list[CriticIssue] = Field(default_factory=list)
    notes: str = ""
