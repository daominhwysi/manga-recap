"""Schemas for Stage 2: Vision page analysis and multi-chapter Story Bible."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PanelAnalysis(BaseModel):
    panel_id: str
    action_summary: str
    dialogue_summary: str = ""
    characters_present: list[str] = Field(default_factory=list)
    emotion: str = "neutral"
    importance: int = Field(default=3, ge=1, le=5)


class PageAnalysis(BaseModel):
    page_id: str
    chapter_id: str
    page_summary: str
    verbatim_quote_vi: str = ""
    visual_punch: str = ""
    panels: list[PanelAnalysis] = Field(default_factory=list)


class CharacterProfile(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = "Nhân vật"
    appearance: str = ""
    personality_notes: str = ""
    relationships: list[str] = Field(default_factory=list)
    first_seen_chapter: str = "ch01"

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "canonical_name" not in d and "name" in d:
            d["canonical_name"] = d.pop("name")
        if isinstance(d.get("relationships"), list):
            rels: list[str] = []
            for item in d["relationships"]:
                if isinstance(item, dict):
                    target = item.get("target", "")
                    rel = item.get("relation", "")
                    rels.append(f"{target}: {rel}".strip(": "))
                else:
                    rels.append(str(item))
            d["relationships"] = rels
        return d


class ChapterSummary(BaseModel):
    chapter_id: str
    title: str = ""
    summary: str = ""
    key_events: list[str] = Field(default_factory=list)
    important_panel_ids: list[str] = Field(default_factory=list)


class StoryBible(BaseModel):
    series_title_guess: str = "Manga Recap"
    genre_guess: str = "Hành động / Giả tưởng"
    tone_summary: str = "Kịch tính, lôi cuốn"
    characters: list[CharacterProfile] = Field(default_factory=list)
    chapter_summaries: list[ChapterSummary] = Field(default_factory=list)
    ongoing_plot_threads: list[str] = Field(default_factory=list)
    pronunciation_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_threads(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if isinstance(d.get("ongoing_plot_threads"), list):
            threads: list[str] = []
            for item in d["ongoing_plot_threads"]:
                if isinstance(item, dict):
                    threads.append(
                        str(item.get("description") or item.get("thread_id") or item)
                    )
                else:
                    threads.append(str(item))
            d["ongoing_plot_threads"] = threads
        raw_pmap = d.get("pronunciation_map")
        if isinstance(raw_pmap, list):
            coerced_map: dict[str, str] = {}
            for item in raw_pmap:
                if isinstance(item, dict):
                    term = str(item.get("term") or item.get("word") or item.get("raw") or "").strip()
                    spoken = str(item.get("spoken") or item.get("phonetic") or item.get("pronunciation") or "").strip()
                    if term and spoken:
                        coerced_map[term] = spoken
            d["pronunciation_map"] = coerced_map
        elif raw_pmap is None:
            d["pronunciation_map"] = {}
        return d
