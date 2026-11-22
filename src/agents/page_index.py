"""Stage 2b: PageIndex Tree Builder (Vectorless hierarchical index + MEMORY.md)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.llm.client import LLMClient
from src.schemas.analysis import (
    ChapterSummary,
    CharacterProfile,
    PageAnalysis,
    StoryBible,
)
from src.schemas.ingestion import JobManifest, PanelInfo


_BACKMATTER_KEYWORDS = (
    "translator's notes",
    "chú thích của dịch giả",
    "ghi chú của dịch giả",
    "afterword",
    "author's notes",
    "lời bạt",
    "hậu ký",
    "tâm sự của tác giả",
    "ghi chú của tác giả",
    "thông tin xuất bản",
    "bản quyền",
    "special thanks",
)

_FRONTMATTER_KEYWORDS = (
    "[phụ lục]",
    "trang trắng",
    "trang lót",
    "mục lục",
    "table of contents",
    "trang bìa",
    "bìa tập",
    "trang tiêu đề",
    "trang mở đầu chương",
    "contents",
)


def _is_backmatter_page(pa: PageAnalysis) -> bool:
    """Detect non-story backmatter pages (credits, translator notes, special thanks)."""
    blob = " ".join(
        [pa.page_summary, *(p.action_summary for p in pa.panels), *(p.dialogue_summary for p in pa.panels)]
    ).lower()
    if blob.startswith("[phụ lục]") or "[phụ lục]" in pa.page_summary.lower():
        return True
    return any(kw in blob for kw in _BACKMATTER_KEYWORDS)


def _is_frontmatter_page(
    pa: PageAnalysis,
    panel_meta: dict[str, PanelInfo] | None = None,
) -> bool:
    """Detect leading cover/blank/title/TOC pages before actual manga storytelling begins.

    Called only while `story` is still empty (leading prefix state), so no arbitrary
    page-index cutoff is needed: as soon as the first real story page appears, frontmatter
    filtering stops automatically.
    """
    blob = " ".join(
        [pa.page_summary, *(p.action_summary for p in pa.panels), *(p.dialogue_summary for p in pa.panels)]
    ).lower()
    if any(
        kw in blob
        for kw in ("[phụ lục]", "trang trắng", "trang lót", "mục lục", "table of contents")
    ):
        return True
    if len(pa.panels) <= 1:
        if any(kw in blob for kw in _FRONTMATTER_KEYWORDS):
            return True
        if pa.panels:
            p0 = pa.panels[0]
            has_dialogue = bool(p0.dialogue_summary.strip())
            info = panel_meta.get(p0.panel_id) if panel_meta else None
            has_text_fp = bool(info and any(fp.kind == "text" for fp in info.focal_points))
            if not has_dialogue and info is not None and not has_text_fp:
                return True
    return False


def _detect_duplicate_spread_pages(manifest: JobManifest) -> set[str]:
    """Identify split portrait pages that duplicate a preceding landscape double-page spread."""
    dup_pages: set[str] = set()
    for ch in manifest.chapters:
        pages = ch.pages
        for idx, pg in enumerate(pages):
            if pg.width > pg.height * 1.15 and idx + 2 < len(pages):
                p1 = pages[idx + 1]
                p2 = pages[idx + 2]
                if not (p1.height > p1.width and p2.height > p2.width):
                    continue
                sp_path = Path(pg.source_path)
                p1_path = Path(p1.source_path)
                p2_path = Path(p2.source_path)
                if not (sp_path.exists() and p1_path.exists() and p2_path.exists()):
                    continue
                img_spread = cv2.imread(str(sp_path), cv2.IMREAD_GRAYSCALE)
                img_p1 = cv2.imread(str(p1_path), cv2.IMREAD_GRAYSCALE)
                img_p2 = cv2.imread(str(p2_path), cv2.IMREAD_GRAYSCALE)
                if img_spread is None or img_p1 is None or img_p2 is None:
                    continue
                w = img_spread.shape[1]
                if w < 64:
                    continue
                left_half = cv2.resize(img_spread[:, : w // 2], (96, 144), interpolation=cv2.INTER_AREA)
                right_half = cv2.resize(img_spread[:, w // 2 :], (96, 144), interpolation=cv2.INTER_AREA)
                r1 = cv2.resize(img_p1, (96, 144), interpolation=cv2.INTER_AREA)
                r2 = cv2.resize(img_p2, (96, 144), interpolation=cv2.INTER_AREA)
                rtl_diff = 0.5 * (
                    float(np.mean(np.abs(right_half.astype(np.float32) - r1.astype(np.float32))))
                    + float(np.mean(np.abs(left_half.astype(np.float32) - r2.astype(np.float32))))
                )
                ltr_diff = 0.5 * (
                    float(np.mean(np.abs(left_half.astype(np.float32) - r1.astype(np.float32))))
                    + float(np.mean(np.abs(right_half.astype(np.float32) - r2.astype(np.float32))))
                )
                if min(rtl_diff, ltr_diff) < 12.0:
                    dup_pages.add(p1.page_id)
                    dup_pages.add(p2.page_id)
    return dup_pages


def _filter_story_pages(
    ch_pages: list[PageAnalysis],
    panel_meta: dict[str, PanelInfo] | None = None,
    duplicate_pages: set[str] | None = None,
) -> tuple[list[PageAnalysis], list[str]]:
    """Separate main story pages from leading frontmatter, duplicate spread splits, and trailing backmatter."""
    if len(ch_pages) <= 5:
        return ch_pages, []

    story: list[PageAnalysis] = []
    skipped: list[str] = []

    for pa in ch_pages:
        low_sum = pa.page_summary.strip().lower()
        if not story and _is_frontmatter_page(pa, panel_meta=panel_meta):
            skipped.append(pa.page_id)
            continue
        if low_sum.startswith("[phụ lục]"):
            skipped.append(pa.page_id)
            continue
        if duplicate_pages and pa.page_id in duplicate_pages:
            skipped.append(pa.page_id)
            continue
        if story and _is_backmatter_page(pa):
            skipped.append(pa.page_id)
            continue
        story.append(pa)

    if not story:
        return ch_pages, []
    return story, skipped


def _extract_page_blocks(md_text: str) -> list[tuple[str, str]]:
    """Split a scene Markdown note into [(page_id, markdown_block), ...]."""
    blocks: list[tuple[str, str]] = []
    cur_pid = ""
    cur_lines: list[str] = []
    for line in md_text.splitlines():
        m = re.match(r"^###\s+([A-Za-z0-9_]+):", line)
        if m:
            if cur_pid and cur_lines:
                blocks.append((cur_pid, "\n".join(cur_lines)))
            cur_pid = m.group(1)
            cur_lines = [line]
        elif cur_pid:
            cur_lines.append(line)
    if cur_pid and cur_lines:
        blocks.append((cur_pid, "\n".join(cur_lines)))
    return blocks


class SceneNode(BaseModel):
    node_id: str
    title: str
    chapter_id: str
    page_range: str
    summary: str
    timeline_beats: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    notes_path: str
    page_ids: list[str] = Field(default_factory=list)
    panel_ids: list[str] = Field(default_factory=list)
    token_est: int = 0


class ArcNode(BaseModel):
    arc_id: str
    title: str
    summary: str
    scene_ids: list[str] = Field(default_factory=list)


class PageIndexTree(BaseModel):
    series_title: str
    total_pages: int
    total_scenes: int
    total_tokens_est: int
    tiers: int = 2  # 2 = Root->Scenes, 3 = Root->Arcs->Scenes
    arcs: list[ArcNode] = Field(default_factory=list)
    scenes: list[SceneNode] = Field(default_factory=list)
    skipped_pages: list[str] = Field(default_factory=list)

    def get_scene(self, node_id: str) -> SceneNode | None:
        for sc in self.scenes:
            if sc.node_id == node_id:
                return sc
        return None

    def get_scene_text(self, node_id: str) -> str:
        sc = self.get_scene(node_id)
        if not sc:
            return ""
        p = Path(sc.notes_path)
        return p.read_text(encoding="utf-8") if p.exists() else sc.summary

    def get_scene_tail_text(self, node_id: str, after_page_id: str) -> tuple[str, list[str]]:
        """Return markdown + panel_ids for pages in `node_id` strictly after `after_page_id`."""
        md = self.get_scene_text(node_id)
        blocks = _extract_page_blocks(md)
        if not blocks:
            return "", []
        page_ids = [pid for pid, _ in blocks]
        if after_page_id not in page_ids:
            return "", []
        cut_idx = page_ids.index(after_page_id) + 1
        if cut_idx >= len(blocks):
            return "", []
        tail_blocks = blocks[cut_idx:]
        tail_text = "\n".join(blk for _, blk in tail_blocks)
        panel_ids = re.findall(r"-\s+\[([A-Za-z0-9_]+)\]", tail_text)
        return tail_text, panel_ids

    def get_all_notes_text(self) -> str:
        chunks: list[str] = []
        for sc in self.scenes:
            p = Path(sc.notes_path)
            if p.exists():
                chunks.append(p.read_text(encoding="utf-8"))
        return "\n\n".join(chunks)

    def compact_outline_view(self) -> str:
        lines: list[str] = [
            f"# TREE INDEX: {self.series_title} ({self.total_pages} story pages, {self.total_scenes} scenes)"
        ]

        def _render_scene(sc: SceneNode) -> list[str]:
            chars = ", ".join(sc.characters[:8])
            out = [
                f"- [{sc.node_id}] ({sc.page_range}) **{sc.title}**: {sc.summary}"
                + (f" | Nhân vật: {chars}" if chars else "")
            ]
            for tb in sc.timeline_beats:
                out.append(f"  * {tb}")
            return out

        if self.tiers == 3 and self.arcs:
            scene_map = {s.node_id: s for s in self.scenes}
            for arc in self.arcs:
                lines.append(f"\n## {arc.arc_id}: {arc.title} — {arc.summary}")
                for sid in arc.scene_ids:
                    sc = scene_map.get(sid)
                    if sc:
                        lines.extend(_render_scene(sc))
        else:
            for sc in self.scenes:
                lines.extend(_render_scene(sc))
        return "\n".join(lines)


class _SceneSummaryLLM(BaseModel):
    title: str
    summary: str
    timeline_beats: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)


class _AliasRule(BaseModel):
    raw_name: str
    canonical_name: str
    page_ids: list[str] = Field(default_factory=list)


class _AliasResolutionResponse(BaseModel):
    rules: list[_AliasRule] = Field(default_factory=list)


def format_pages_compact_md(
    pages: list[PageAnalysis],
    panel_meta: dict[str, PanelInfo] | None = None,
) -> str:
    """Convert PageAnalysis list to ultra-compact Markdown preserving every panel_id + visual tags."""
    lines: list[str] = []
    for pa in pages:
        lines.append(f"### {pa.page_id}: {pa.page_summary}")
        for p in pa.panels:
            dlg = f' | "{p.dialogue_summary}"' if p.dialogue_summary else ""
            chars = f" [{', '.join(p.characters_present)}]" if p.characters_present else ""
            vis_tag = ""
            if panel_meta and p.panel_id in panel_meta:
                info = panel_meta[p.panel_id]
                ar = round(info.width / max(1, info.height), 2)
                has_face = any(fp.kind == "face" for fp in info.focal_points)
                has_text = any(fp.kind == "text" for fp in info.focal_points) or bool(p.dialogue_summary)
                flags = "+".join(
                    f for f, ok in (("face", has_face), ("text", has_text)) if ok
                ) or "bg-only"
                vis_tag = f", ar={ar}, {flags}"
            lines.append(
                f"- [{p.panel_id}] ({p.emotion}, imp={p.importance}{vis_tag}){chars} {p.action_summary}{dlg}"
            )
    return "\n".join(lines)


def render_memory_md(bible: StoryBible, tree: PageIndexTree) -> str:
    """Render MEMORY.md with canonical character profiles, origin nodes, and plot threads."""
    char_origin: dict[str, str] = {}
    for sc in tree.scenes:
        for c in sc.characters:
            char_origin.setdefault(c.lower(), sc.node_id)

    lines = [
        f"# MEMORY: {bible.series_title_guess}",
        f"- Thể loại: {bible.genre_guess}",
        f"- Giọng điệu: {bible.tone_summary}",
        "- Lưu ý chống nhầm tên: Nếu dữ liệu từng trang ghi mô tả ngoại hình chung chung hoặc nhận nhầm nhân vật, hãy đối chiếu ngoại hình & ngữ cảnh với Bảng Nhân Vật Chuẩn bên dưới để gọi đúng tên.",
        "",
        "## Nhân vật chuẩn (Canonical Cast)",
    ]
    for ch in bible.characters:
        aliases = f" (bí danh: {', '.join(ch.aliases)})" if ch.aliases else ""
        origin = char_origin.get(ch.canonical_name.lower(), ch.first_seen_chapter)
        rels = f" | Quan hệ: {'; '.join(ch.relationships)}" if ch.relationships else ""
        lines.append(
            f"- **{ch.canonical_name}**{aliases} [{ch.role}, gốc: {origin}]: "
            f"{ch.appearance} {ch.personality_notes}{rels}"
        )

    if bible.ongoing_plot_threads:
        lines.extend(["", "## Mạch bí ẩn & Plot Threads"])
        for pt in bible.ongoing_plot_threads:
            lines.append(f"- {pt}")

    return "\n".join(lines)


class PageIndexBuilder:
    """Builds hierarchical PageIndex tree, compact Markdown notes, and StoryBible/MEMORY.md."""

    def __init__(self, llm: LLMClient, pages_per_scene: int = 20) -> None:
        self.llm = llm
        self.pages_per_scene = max(5, pages_per_scene)

    def _resolve_character_aliases(
        self,
        page_analyses: list[PageAnalysis],
        job_dir: Path,
    ) -> list[PageAnalysis]:
        """Post-2a Alias Resolution: unify variants & '[Chưa rõ tên: ...]' tags against seed_cast."""
        if not self.llm.has_text_credentials or len(page_analyses) <= 8:
            return page_analyses

        seed_path = job_dir / "repo" / "seed_cast.json"
        seed_info = ""
        if seed_path.exists():
            try:
                seed_info = seed_path.read_text(encoding="utf-8")[:6000]
            except Exception:
                seed_info = ""

        # Collect distinct labels with page occurrences
        label_pages: dict[str, list[str]] = {}
        unknown_contexts: list[str] = []
        for pa in page_analyses:
            seen_on_page: set[str] = set()
            for pnl in pa.panels:
                for ch in pnl.characters_present:
                    lbl = ch.strip()
                    if not lbl:
                        continue
                    label_pages.setdefault(lbl, [])
                    if pa.page_id not in label_pages[lbl]:
                        label_pages[lbl].append(pa.page_id)
                    if (
                        lbl not in seen_on_page
                        and (
                            lbl.startswith("[")
                            or "chưa rõ" in lbl.lower()
                            or "bí ẩn" in lbl.lower()
                            or "người lạ" in lbl.lower()
                        )
                    ):
                        seen_on_page.add(lbl)
                        unknown_contexts.append(f"- {pa.page_id} [{lbl}]: {pa.page_summary[:180]}")

        label_summary = "\n".join(
            f"- {lbl!r} (xuất hiện ở {len(pgs)} trang: {pgs[:6]})"
            for lbl, pgs in sorted(label_pages.items(), key=lambda kv: -len(kv[1]))[:45]
        )
        unk_block = "\n".join(unknown_contexts[:30])

        try:
            res = self.llm.complete_json(
                agent_name="AliasResolverAgent",
                system_prompt=(
                    "Bạn là biên tập viên chuẩn hóa danh tính nhân vật trước khi lập mục lục.\n"
                    "Nhiệm vụ: gom các biến thể tên gọi (tên tắt / họ tên đầy đủ) về đúng `canonical_name` "
                    "CHỈ KHI có đủ bằng chứng trong seed_cast và tóm tắt trang. "
                    "Tuyệt đối KHÔNG gộp nhầm 2 nhân vật khác nhau và KHÔNG gán tên thật toàn cục cho các nhãn mô tả tạm "
                    "'[Chưa rõ tên: ...]' ở những trang nhân vật chưa lộ danh tính. "
                    "Nếu nhãn chỉ rõ ở một số trang cụ thể, điền danh sách `page_ids`; nếu áp dụng toàn tập, để `page_ids` trống []."
                ),
                user_prompt=(
                    f"Seed cast:\n{seed_info or 'Không có'}\n\n"
                    f"Các nhãn nhân vật xuất hiện trong 02a:\n{label_summary}\n\n"
                    f"Ngữ cảnh các nhãn chưa rõ tên:\n{unk_block}"
                ),
                schema_cls=_AliasResolutionResponse,
            )
        except Exception:
            return page_analyses

        global_map: dict[str, str] = {}
        scoped_map: dict[tuple[str, str], str] = {}
        for rule in res.rules:
            raw = rule.raw_name.strip()
            canon = rule.canonical_name.strip()
            if not raw or not canon:
                continue
            if rule.page_ids:
                for pid in rule.page_ids:
                    scoped_map[(raw.lower(), pid)] = canon
            elif not raw.startswith("[") and "chưa rõ" not in raw.lower() and "bí ẩn" not in raw.lower():
                global_map[raw.lower()] = canon

        if not global_map and not scoped_map:
            return page_analyses

        resolved: list[PageAnalysis] = []
        for pa in page_analyses:
            pa_copy = pa.model_copy(deep=True)
            for pnl in pa_copy.panels:
                new_chars: list[str] = []
                for ch in pnl.characters_present:
                    key = ch.strip().lower()
                    canon = scoped_map.get((key, pa_copy.page_id)) or global_map.get(key) or ch.strip()
                    if canon and canon not in new_chars:
                        new_chars.append(canon)
                pnl.characters_present = new_chars
            resolved.append(pa_copy)
        return resolved

    def build(
        self,
        manifest: JobManifest,
        page_analyses: list[PageAnalysis],
        job_dir: Path,
        force: bool = False,
    ) -> tuple[PageIndexTree, StoryBible]:
        repo_dir = job_dir / "repo"
        notes_dir = repo_dir / "notes"
        tree_path = repo_dir / "tree_index.json"
        memory_path = repo_dir / "MEMORY.md"
        bible_path = job_dir / "02b_story_bible.json"

        if tree_path.exists() and bible_path.exists() and memory_path.exists() and not force:
            tree = PageIndexTree.model_validate_json(tree_path.read_text(encoding="utf-8"))
            bible = StoryBible.model_validate_json(bible_path.read_text(encoding="utf-8"))
            return tree, bible

        notes_dir.mkdir(parents=True, exist_ok=True)
        panel_meta: dict[str, PanelInfo] = {
            pnl.panel_id: pnl
            for ch in manifest.chapters
            for pg in ch.pages
            for pnl in pg.panels
        }
        resolved_analyses = self._resolve_character_aliases(page_analyses, job_dir)

        skipped_all: list[str] = []
        dup_spread_pages = _detect_duplicate_spread_pages(manifest)
        total_tokens = 0
        total_story_pages = 0

        by_ch: dict[str, list[PageAnalysis]] = {}
        for pa in resolved_analyses:
            by_ch.setdefault(pa.chapter_id, []).append(pa)

        # Prepare all scene chunks first so SceneSummarizer can run in parallel
        chunk_jobs: list[dict] = []
        scene_counter = 1
        for ch in manifest.chapters:
            raw_ch_pages = by_ch.get(ch.chapter_id, [])
            ch_pages, ch_skipped = _filter_story_pages(
                raw_ch_pages,
                panel_meta=panel_meta,
                duplicate_pages=dup_spread_pages,
            )
            skipped_all.extend(ch_skipped)
            total_story_pages += len(ch_pages)

            for start in range(0, len(ch_pages), self.pages_per_scene):
                chunk = ch_pages[start : start + self.pages_per_scene]
                node_id = f"scene_{scene_counter:03d}"
                scene_counter += 1
                page_range = f"{chunk[0].page_id}..{chunk[-1].page_id}"
                md_body = format_pages_compact_md(chunk, panel_meta=panel_meta)
                note_file = notes_dir / f"{node_id}.md"
                note_file.write_text(
                    f"# {node_id} ({ch.chapter_id}, {page_range})\n\n{md_body}\n",
                    encoding="utf-8",
                )
                tok_est = max(1, len(md_body) // 4)
                total_tokens += tok_est
                chunk_jobs.append(
                    {
                        "node_id": node_id,
                        "chapter": ch,
                        "chunk": chunk,
                        "page_range": page_range,
                        "md_body": md_body,
                        "note_file": note_file,
                        "tok_est": tok_est,
                    }
                )

        def _build_scene_node(job: dict) -> SceneNode:
            node_id = job["node_id"]
            ch = job["chapter"]
            chunk: list[PageAnalysis] = job["chunk"]
            page_range = job["page_range"]
            md_body = job["md_body"]
            note_file = job["note_file"]
            tok_est = job["tok_est"]

            pg_ids = [pa.page_id for pa in chunk]
            pids = [pnl.panel_id for pa in chunk for pnl in pa.panels]

            chars_seen: list[str] = []
            for pa in chunk:
                for pnl in pa.panels:
                    for c in pnl.characters_present:
                        if c and c not in chars_seen:
                            chars_seen.append(c)

            timeline_beats: list[str] = []
            if self.llm.has_text_credentials and len(page_analyses) > 30:
                try:
                    sc_sum = self.llm.complete_json(
                        agent_name="PageIndexSceneSummarizer",
                        system_prompt=(
                            "Bạn lập mục lục phân đoạn manga đầy đủ từ đầu đến cuối phân đoạn.\n"
                            "BẮT BUỘC:\n"
                            "1) `summary`: tóm tắt tổng quan 1-2 câu (40-70 từ) phủ trọn từ trang đầu tới trang cuối.\n"
                            "2) `timeline_beats`: danh sách 4-5 gạch đầu dòng theo đúng thứ tự thời gian kèm mốc trang "
                            "(ví dụ: '[p101-p105] ...', '[p106-p110] ...', '[p111-p118] ...', '[p119-p120] ...'), "
                            "TUYỆT ĐỐI KHÔNG bỏ sót các sự kiện ở nửa cuối và trang cuối của phân đoạn."
                        ),
                        user_prompt=f"Đoạn {node_id} ({page_range}):\n{md_body[:24000]}",
                        schema_cls=_SceneSummaryLLM,
                    )
                    title = sc_sum.title
                    summary = sc_sum.summary
                    timeline_beats = sc_sum.timeline_beats[:6]
                    if sc_sum.characters:
                        chars_seen = sc_sum.characters
                except Exception:
                    title = f"{ch.title} ({page_range})"
                    summary = f"{chunk[0].page_summary} ... {chunk[-1].page_summary}"[:260]
            else:
                title = f"{ch.title} ({page_range})"
                summary = " | ".join(pa.page_summary for pa in chunk[:2])[:220]
                if len(chunk) >= 4:
                    timeline_beats = [
                        f"[{chunk[0].page_id}] {chunk[0].page_summary}",
                        f"[{chunk[len(chunk) // 2].page_id}] {chunk[len(chunk) // 2].page_summary}",
                        f"[{chunk[-1].page_id}] {chunk[-1].page_summary}",
                    ]

            return SceneNode(
                node_id=node_id,
                title=title,
                chapter_id=ch.chapter_id,
                page_range=page_range,
                summary=summary,
                timeline_beats=timeline_beats,
                characters=chars_seen[:12],
                notes_path=str(note_file.resolve()),
                page_ids=pg_ids,
                panel_ids=pids,
                token_est=tok_est,
            )

        if self.llm.has_text_credentials and len(page_analyses) > 30 and len(chunk_jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(chunk_jobs))) as pool:
                scenes = list(pool.map(_build_scene_node, chunk_jobs))
        else:
            scenes = [_build_scene_node(j) for j in chunk_jobs]

        # Build 3-tier arcs automatically if > 40 scenes (for 1,000+ page scale)
        arcs: list[ArcNode] = []
        tiers = 2
        if len(scenes) > 40:
            tiers = 3
            arc_size = 12
            for a_idx in range(0, len(scenes), arc_size):
                grp = scenes[a_idx : a_idx + arc_size]
                arcs.append(
                    ArcNode(
                        arc_id=f"arc_{len(arcs) + 1:02d}",
                        title=f"Phần {len(arcs) + 1} ({grp[0].page_range.split('..')[0]}..{grp[-1].page_range.split('..')[-1]})",
                        summary=" / ".join(s.title for s in grp[:3]),
                        scene_ids=[s.node_id for s in grp],
                    )
                )

        tree = PageIndexTree(
            series_title=manifest.job_id,
            total_pages=total_story_pages or len(page_analyses),
            total_scenes=len(scenes),
            total_tokens_est=total_tokens,
            tiers=tiers,
            arcs=arcs,
            scenes=scenes,
            skipped_pages=skipped_all,
        )

        # Build StoryBible from Scene timeline summaries (compact, covers entire story cleanly)
        if self.llm.has_text_credentials:
            bible = self.llm.complete_json(
                agent_name="StoryBibleAgent",
                system_prompt=(
                    "Bạn là StoryBibleAgent quản lý hồ sơ nhân vật chuẩn (canonical_name, aliases, role, "
                    "appearance, personality_notes, relationships) và mạch cốt truyện xuyên suốt bộ truyện."
                ),
                user_prompt=(
                    f"Series: {manifest.job_id}\n"
                    f"Mục lục chi tiết từng phân đoạn:\n{tree.compact_outline_view()}\n\n"
                    "Hãy tổng hợp StoryBible hoàn chỉnh dạng JSON (series_title_guess, genre_guess, tone_summary, "
                    "characters, chapter_summaries, ongoing_plot_threads)."
                ),
                schema_cls=StoryBible,
            )
        else:
            all_chars: list[CharacterProfile] = []
            seen_names: set[str] = set()
            for sc in scenes:
                for cname in sc.characters:
                    if cname.lower() not in seen_names:
                        seen_names.add(cname.lower())
                        all_chars.append(
                            CharacterProfile(canonical_name=cname, first_seen_chapter=sc.chapter_id)
                        )
            bible = StoryBible(
                series_title_guess=manifest.job_id,
                characters=all_chars[:20],
                chapter_summaries=[
                    ChapterSummary(
                        chapter_id=ch.chapter_id,
                        title=ch.title,
                        summary=f"Tóm tắt {ch.title} ({len(by_ch.get(ch.chapter_id, []))} trang).",
                        important_panel_ids=[],
                    )
                    for ch in manifest.chapters
                ],
            )

        tree_path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        memory_path.write_text(render_memory_md(bible, tree), encoding="utf-8")
        bible_path.write_text(bible.model_dump_json(indent=2), encoding="utf-8")
        return tree, bible
