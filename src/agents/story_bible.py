"""Stage 2b Agent: Maintains multi-chapter Story Bible and character consistency."""

from __future__ import annotations

import json
from pathlib import Path

from src.llm.client import LLMClient
from src.schemas.analysis import ChapterSummary, PageAnalysis, StoryBible
from src.schemas.ingestion import JobManifest


SYSTEM_PROMPT = """Bạn là StoryBibleAgent quản lý hồ sơ nhân vật và mạch cốt truyện xuyên suốt nhiều chương manga.
Nhiệm vụ: hợp nhất ghi chú các trang thành bản cập nhật StoryBible nhất quán tên nhân vật, quan hệ và các sự kiện quan trọng."""


class StoryBibleAgent:
    """Builds and updates StoryBible across chapters."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def build(
        self,
        manifest: JobManifest,
        page_analyses: list[PageAnalysis],
        job_dir: Path,
        force: bool = False,
    ) -> StoryBible:
        bible_path = job_dir / "02b_story_bible.json"
        if bible_path.exists() and not force:
            return StoryBible.model_validate_json(bible_path.read_text(encoding="utf-8"))

        chapters_dir = job_dir / "02b_chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        by_chapter: dict[str, list[PageAnalysis]] = {}
        for pa in page_analyses:
            by_chapter.setdefault(pa.chapter_id, []).append(pa)

        bible = StoryBible(series_title_guess=manifest.job_id)

        for ch in manifest.chapters:
            ch_pages = by_chapter.get(ch.chapter_id, [])
            if self.llm.has_text_credentials:
                user_prompt = (
                    f"StoryBible hiện tại:\n{bible.model_dump_json(indent=2)}\n\n"
                    f"Dữ liệu chương mới ({ch.chapter_id} - {ch.title}):\n"
                    f"{json.dumps([p.model_dump() for p in ch_pages], ensure_ascii=False)}\n\n"
                    "Hãy cập nhật và trả về toàn bộ StoryBible mới dạng JSON."
                )
                bible = self.llm.complete_json(
                    agent_name="StoryBibleAgent",
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema_cls=StoryBible,
                )
            elif self.llm.config.offline_fallback:
                important_ids = [
                    pnl.panel_id
                    for pa in ch_pages
                    for pnl in pa.panels
                    if pnl.importance >= 4
                ]
                if not important_ids:
                    important_ids = [pnl.panel_id for pa in ch_pages for pnl in pa.panels[:2]]
                summary = ChapterSummary(
                    chapter_id=ch.chapter_id,
                    title=ch.title,
                    summary=f"Tóm tắt diễn biến chính trong {ch.title} qua {len(ch_pages)} trang.",
                    key_events=[pa.page_summary for pa in ch_pages[:4]],
                    important_panel_ids=important_ids[:12],
                )
                bible.chapter_summaries.append(summary)
            else:
                raise RuntimeError("No text API credentials available for StoryBibleAgent.")

            ch_file = chapters_dir / f"{ch.chapter_id}_bible.json"
            ch_file.write_text(bible.model_dump_json(indent=2), encoding="utf-8")

        bible_path.write_text(bible.model_dump_json(indent=2), encoding="utf-8")
        return bible
