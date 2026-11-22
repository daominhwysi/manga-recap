"""Schemas for Stage 1: Ingestion and CV panel detection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class FocalPoint(BaseModel):
    kind: str  # face, body, text, action
    bbox: BBox
    norm_cx: float
    norm_cy: float
    norm_w: float
    norm_h: float
    confidence: float = 1.0


class PanelInfo(BaseModel):
    panel_id: str
    chapter_id: str
    page_id: str
    index_in_page: int
    global_index: int
    bbox: BBox
    width: int
    height: int
    crop_path: str
    focal_points: list[FocalPoint] = Field(default_factory=list)


class PageInfo(BaseModel):
    page_id: str
    chapter_id: str
    page_index: int
    source_path: str
    format: Literal["manga", "manhwa"] = "manga"
    width: int
    height: int
    som_paths: list[str] = Field(default_factory=list)
    panels: list[PanelInfo] = Field(default_factory=list)


class ChapterInfo(BaseModel):
    chapter_id: str
    title: str
    pages: list[PageInfo] = Field(default_factory=list)


class JobManifest(BaseModel):
    job_id: str
    input_path: str
    format_mode: str = "auto"
    reading_order: str = "rtl"
    chapters: list[ChapterInfo] = Field(default_factory=list)
