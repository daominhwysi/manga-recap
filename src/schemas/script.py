"""Schemas for Stage 3: Narrative Scriptwriter and Critic review."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

VALID_MOODS = ("calm", "warm", "comedic", "mysterious", "tense", "action", "tragic")

_MOOD_SYNONYM_MAP: dict[str, str] = {
    "calm": "calm",
    "peaceful": "calm",
    "quiet": "calm",
    "neutral": "calm",
    "melancholic": "calm",
    "warm": "warm",
    "gentle": "warm",
    "heartwarming": "warm",
    "happy": "warm",
    "hopeful": "warm",
    "relieved": "warm",
    "tender": "warm",
    "comedic": "comedic",
    "comedy": "comedic",
    "funny": "comedic",
    "playful": "comedic",
    "humorous": "comedic",
    "sarcastic": "comedic",
    "lighthearted": "comedic",
    "amused": "comedic",
    "mysterious": "mysterious",
    "mystery": "mysterious",
    "curious": "mysterious",
    "suspense": "mysterious",
    "suspenseful": "mysterious",
    "eerie": "mysterious",
    "ominous": "mysterious",
    "intriguing": "mysterious",
    "uncanny": "mysterious",
    "tense": "tense",
    "dramatic": "tense",
    "shocking": "tense",
    "shocked": "tense",
    "urgent": "tense",
    "anxious": "tense",
    "fearful": "tense",
    "terrifying": "tense",
    "horrific": "tense",
    "horror": "tense",
    "creepy": "tense",
    "dark": "tense",
    "action": "action",
    "explosive": "action",
    "violent": "action",
    "intense": "action",
    "chaotic": "action",
    "epic": "action",
    "fierce": "action",
    "climax": "action",
    "battle": "action",
    "combat": "action",
    "tragic": "tragic",
    "sad": "tragic",
    "sorrow": "tragic",
    "grief": "tragic",
    "desperate": "tragic",
    "heartbreaking": "tragic",
    "emotional": "tragic",
    "painful": "tragic",
}


def normalize_mood(raw: Any) -> str:
    """Coerce free-form LLM mood descriptors into the canonical 7-mood contract."""
    if not raw:
        return "tense"
    token = str(raw).strip().lower()
    if token in _MOOD_SYNONYM_MAP:
        return _MOOD_SYNONYM_MAP[token]
    for key, mapped in _MOOD_SYNONYM_MAP.items():
        if key in token:
            return mapped
    return "tense"


class ScriptBeat(BaseModel):
    beat_id: str
    section: str = "setup"  # hook, setup, rising, climax, cliffhanger
    beat_type: str = "narrative"  # hook_montage, narrative, commentary, outro
    narration_vi: str
    tts_text: str = ""
    panel_ids: list[str] = Field(default_factory=list)
    mood: str = "tense"
    pause_after_ms: int = 240

    @field_validator("mood", mode="before")
    @classmethod
    def _coerce_mood(cls, v: Any) -> str:
        return normalize_mood(v)


class RecapScript(BaseModel):
    title_vi: str
    persona: str = "kich_tinh"
    hook_strategy: str = "in_medias_res"  # in_medias_res, montage_tease, world_premise
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
