"""Stage 1.5 + 2a Agent: Pre-Pass Cast Discovery (cast_sheet.jpg) + Chunked Parallel Rolling Vision."""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.llm.client import LLMClient
from src.schemas.analysis import PageAnalysis, PanelAnalysis
from src.schemas.ingestion import JobManifest, PageInfo, PanelInfo

SYSTEM_PROMPT = """Bạn là PageAnalyzerAgent chuyên đọc trang manga/manhwa đã đánh dấu Set-of-Mark ([P1], [P2]...).
Quy tắc bắt buộc:
1) Viết 100% bằng TIẾNG VIỆT tự nhiên, chính xác (giữ nguyên tên riêng nhân vật, địa danh, tổ chức và thuật ngữ đặc trưng theo bản gốc).
2) Phân tích từng khung hình theo đúng thứ tự panel_id từ [P1] đến [Pn] theo chiều đọc.
3) Nếu đã đính kèm ảnh BẢNG CHÂN DUNG NHÂN VẬT (cast_sheet.jpg), BẮT BUỘC đối chiếu gương mặt, kiểu tóc và trang phục với bảng đó kết hợp với Ngữ cảnh cảnh trước (Rolling Scene State) để gọi ĐÚNG tên nhân vật, tránh nhầm lẫn nhân vật này sang nhân vật khác.
4) Trong `dialogue_summary`, tóm tắt hoặc trích dịch gọn lời thoại quan trọng nhất trong bóng thoại bằng tiếng Việt rõ nghĩa.
5) Nếu trang là trang bìa (cover), trang trắng/trang lót, trang tiêu đề (title page), mục lục (table of contents), lời cảm ơn (Special Thanks), chú thích dịch giả (Translator's Notes), trang bản quyền hay truyện ngắn tặng kèm cuối tập, BẮT BUỘC ghi rõ cụm "[PHỤ LỤC]" ở đầu `page_summary`.
6) CHỐNG BỊA TÊN (BẮT BUỘC): Chỉ dùng tên nhân vật có trong Bảng Chân Dung / Rolling Scene State hoặc được gọi rõ trong bóng thoại/chữ trên trang. TUYỆT ĐỐI KHÔNG tự đoán tên từ kiến thức ngoài truyện khi tên đó chưa hề xuất hiện; nếu là người lạ chưa xưng tên, hãy ghi dạng `[Chưa rõ tên: đặc điểm ngoại hình nổi bật]` (ví dụ `[Chưa rõ tên: người đeo mặt nạ mặc áo choàng tối màu]`).
7) CHỐNG BỊA CỐT TRUYỆN TRÊN TRANG BÌA / TRANG TRẮNG / MỤC LỤC (BẮT BUỘC): Nếu trang chỉ là trang trắng, trang bìa, trang tiêu đề hoặc mục lục (không có bóng thoại truyện), chỉ mô tả đúng những gì in trên trang (ví dụ: `[PHỤ LỤC] Trang trắng`, `[PHỤ LỤC] Trang mục lục tập 1`). TUYỆT ĐỐI KHÔNG lấy trí nhớ về Anime hay các trang sau để bịa ra cảnh hành động hoặc lời thoại không có trên ảnh."""


class PageBatchAnalysis(BaseModel):
    pages: list[PageAnalysis] = Field(default_factory=list)


class SeedCharacter(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    appearance_vi: str = ""
    best_panel_id: str = ""


class PrePassCastResult(BaseModel):
    characters: list[SeedCharacter] = Field(default_factory=list)


def _is_blank_or_solid_page(image_path: str | Path) -> bool:
    """Detect nearly blank/solid filler pages so Vision never hallucinates plot onto them."""
    p = Path(image_path)
    if not p.exists() or p.stat().st_size < 128:
        return False
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    return float(np.std(img)) < 4.5


def _make_blank_page_analysis(page: PageInfo) -> PageAnalysis:
    """Deterministic analysis for blank/filler pages with no manga panels/text."""
    return PageAnalysis(
        page_id=page.page_id,
        chapter_id=page.chapter_id,
        page_summary="[PHỤ LỤC] Trang trắng (trang lót không có nội dung truyện).",
        panels=[
            PanelAnalysis(
                panel_id=p.panel_id,
                action_summary="[PHỤ LỤC] Trang trắng.",
                dialogue_summary="",
                characters_present=[],
                emotion="neutral",
                importance=1,
            )
            for p in page.panels
        ],
    )


def _is_dummy_page_analysis(pa: PageAnalysis) -> bool:
    """Detect synthetic offline fallback placeholders so they are never kept as valid Vision cache."""
    if re.match(r"^Trang\s+\d+\s+của\s+\S+\s+gồm\s+\d+\s+khung hình\.?$", pa.page_summary.strip()):
        return True
    if pa.panels and all("diễn biến cảnh truyện." in p.action_summary for p in pa.panels):
        return True
    return False


def _select_prepass_pages(manifest: JobManifest, max_samples: int = 16) -> list[PageInfo]:
    """Select face-rich, dialogue-rich pages distributed across the volume for Pre-Pass Cast Discovery."""
    all_pages = [pg for ch in manifest.chapters for pg in ch.pages]
    if len(all_pages) <= max_samples:
        return all_pages

    scored: list[tuple[float, int, PageInfo]] = []
    for idx, pg in enumerate(all_pages):
        if _is_blank_or_solid_page(pg.source_path):
            continue
        faces = sum(1 for pnl in pg.panels for fp in pnl.focal_points if fp.kind == "face")
        texts = sum(1 for pnl in pg.panels for fp in pnl.focal_points if fp.kind == "text")
        bodies = sum(1 for pnl in pg.panels for fp in pnl.focal_points if fp.kind == "body")
        score = faces * 3.0 + texts * 2.2 + bodies * 0.6
        if faces == 0 and texts == 0:
            score *= 0.1
        scored.append((score, idx, pg))

    if not scored:
        return all_pages[:max_samples]

    num_buckets = 6
    bucket_size = max(1, len(all_pages) // num_buckets)
    selected_idx: set[int] = set()

    for b in range(num_buckets):
        lo = b * bucket_size
        hi = len(all_pages) if b == num_buckets - 1 else (b + 1) * bucket_size
        bucket_items = sorted(
            [item for item in scored if lo <= item[1] < hi],
            key=lambda x: x[0],
            reverse=True,
        )
        for _, idx, _ in bucket_items[:2]:
            selected_idx.add(idx)

    # Top up remaining slots with highest overall scores
    for _, idx, _ in sorted(scored, key=lambda x: x[0], reverse=True):
        if len(selected_idx) >= max_samples:
            break
        selected_idx.add(idx)

    return [all_pages[i] for i in sorted(selected_idx)[:max_samples]]


def _crop_face_portrait(panel: PanelInfo, target_w: int = 188, target_h: int = 192) -> np.ndarray | None:
    """Crop the clearest face (+35-50% margin for hair & collar) from a panel image."""
    img = cv2.imread(panel.crop_path)
    if img is None:
        return None
    ph, pw = img.shape[:2]
    face_fps = [fp for fp in panel.focal_points if fp.kind == "face"]
    if face_fps:
        # Prefer largest high-confidence face bbox inside the panel
        best_fp = max(face_fps, key=lambda fp: fp.confidence * max(0.01, fp.norm_w * fp.norm_h))
        b = best_fp.bbox
        bw = max(24, b.width)
        bh = max(24, b.height)
        x1 = max(0, int(b.x1 - 0.45 * bw))
        x2 = min(pw, int(b.x2 + 0.45 * bw))
        y1 = max(0, int(b.y1 - 0.55 * bh))
        y2 = min(ph, int(b.y2 + 0.40 * bh))
        crop = img[y1:y2, x1:x2]
    else:
        # Fallback: upper-center portrait crop
        x1, x2 = int(pw * 0.1), int(pw * 0.9)
        y1, y2 = 0, min(ph, int(pw * 1.15))
        crop = img[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else img

    if crop.size == 0:
        crop = img
    ch, cw = crop.shape[:2]
    scale = min(target_w / max(1, cw), target_h / max(1, ch))
    nw, nh = max(24, int(round(cw * scale))), max(24, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    tile = np.full((target_h, target_w, 3), 24, dtype=np.uint8)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    tile[oy : oy + nh, ox : ox + nw] = resized
    return tile


def _build_cast_sheet(
    seed_cast: list[SeedCharacter],
    manifest: JobManifest,
    out_path: Path,
    max_cards: int = 12,
) -> Path | None:
    """Assemble discovered character portraits into a single labeled cast_sheet.jpg reference grid."""
    panel_map: dict[str, PanelInfo] = {
        pnl.panel_id: pnl
        for ch in manifest.chapters
        for pg in ch.pages
        for pnl in pg.panels
    }
    page_by_panel: dict[str, PageInfo] = {
        pnl.panel_id: pg
        for ch in manifest.chapters
        for pg in ch.pages
        for pnl in pg.panels
    }

    cards: list[np.ndarray] = []
    card_w, card_h = 200, 232
    tile_w, tile_h = 188, 192

    for ch_entry in seed_cast[:max_cards]:
        pnl = panel_map.get(ch_entry.best_panel_id)
        if pnl is None:
            continue
        # If chosen panel has >1 faces, check same page for a clean single-face panel first
        face_count = sum(1 for fp in pnl.focal_points if fp.kind == "face")
        if face_count > 1 and ch_entry.best_panel_id in page_by_panel:
            pg = page_by_panel[ch_entry.best_panel_id]
            single_face_pnls = [
                cand
                for cand in pg.panels
                if sum(1 for fp in cand.focal_points if fp.kind == "face") == 1
            ]
            if len(single_face_pnls) == 1:
                pnl = single_face_pnls[0]

        portrait = _crop_face_portrait(pnl, target_w=tile_w, target_h=tile_h)
        if portrait is None:
            continue

        card = np.full((card_h, card_w, 3), 18, dtype=np.uint8)
        card[6 : 6 + tile_h, 6 : 6 + tile_w] = portrait
        cv2.rectangle(card, (5, 5), (6 + tile_w, 6 + tile_h), (210, 210, 210), 1)
        label = ch_entry.canonical_name[:22]
        cv2.putText(
            card,
            label,
            (8, card_h - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cards.append(card)

    if not cards:
        return None

    cols = min(4, max(2, math.ceil(math.sqrt(len(cards)))))
    rows = math.ceil(len(cards) / cols)
    pad = 8
    grid_w = cols * card_w + (cols + 1) * pad
    grid_h = rows * card_h + (rows + 1) * pad + 28
    canvas = np.full((grid_h, grid_w, 3), 10, dtype=np.uint8)
    cv2.putText(
        canvas,
        "CAST REFERENCE SHEET (match faces & hairstyles strictly)",
        (pad, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (120, 230, 255),
        1,
        cv2.LINE_AA,
    )
    for idx, card in enumerate(cards):
        r = idx // cols
        c = idx % cols
        x = pad + c * (card_w + pad)
        y = 28 + pad + r * (card_h + pad)
        canvas[y : y + card_h, x : x + card_w] = card

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out_path


class PageAnalyzerAgent:
    """Extracts grounded per-panel actions & dialogues using Pre-Pass Cast Sheet + Chunked Parallel Rolling."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def discover_seed_cast(
        self,
        manifest: JobManifest,
        job_dir: Path,
        force: bool = False,
        max_workers: int = 4,
    ) -> tuple[list[SeedCharacter], Path | None]:
        """Stage 1.5 Pre-Pass: scan face/dialogue-rich sample pages to lock cast names + build cast_sheet.jpg."""
        all_pages = [pg for ch in manifest.chapters for pg in ch.pages]
        repo_dir = job_dir / "repo"
        seed_path = repo_dir / "seed_cast.json"
        sheet_path = repo_dir / "cast_sheet.jpg"

        if seed_path.exists() and not force:
            try:
                cached = PrePassCastResult.model_validate_json(seed_path.read_text(encoding="utf-8"))
                return cached.characters, (sheet_path if sheet_path.exists() else None)
            except Exception:
                pass

        if len(all_pages) < 8 or not self.llm.has_vision_credentials:
            return [], None

        samples = _select_prepass_pages(manifest, max_samples=16)
        batches = [samples[i : i + 4] for i in range(0, len(samples), 4)]

        def _scan_sample_batch(batch: list[PageInfo]) -> list[SeedCharacter]:
            imgs = [Path(p.som_paths[0]) if p.som_paths else Path(p.source_path) for p in batch]
            specs = "\n".join(
                f"- Ảnh #{i}: page_id={p.page_id}, panels=[{', '.join(pnl.panel_id for pnl in p.panels)}]"
                for i, p in enumerate(batch, start=1)
            )
            prompt = (
                "Quét nhanh các trang mẫu sau để lập DANH SÁCH NHÂN VẬT CÓ THẬT TRONG TẬP TRUYỆN.\n"
                "QUY TẮC CỨNG CHỐNG BỊA TÊN:\n"
                "1) CHỈ đưa vào danh sách nhân vật có TÊN ĐƯỢC GỌI RÕ trong bóng thoại/chú thích/bìa của các trang này "
                "(hoặc nhân vật chính/quan trọng lộ mặt rõ với mô tả ngoại hình cụ thể).\n"
                "2) Bắt buộc điền `evidence_quote` là câu thoại/chữ gốc trên trang nhắc tới tên đó; TUYỆT ĐỐI KHÔNG tự bịa tên từ kiến thức ngoài truyện.\n"
                "3) `best_panel_id`: chọn đúng 1 panel_id nơi nhân vật lộ mặt to, rõ nhất (ưu tiên panel chỉ có 1 người).\n\n"
                f"Các trang đính kèm:\n{specs}\n"
                "Trả về JSON PrePassCastResult gồm `characters` (canonical_name, aliases, evidence_quote, appearance_vi, best_panel_id)."
            )
            try:
                res = self.llm.complete_json(
                    agent_name="PrePassCastScanner",
                    system_prompt=(
                        "Bạn là chuyên gia nhận diện nhân vật manga đen trắng. "
                        "Chỉ trích xuất tên nhân vật thực sự xuất hiện trên trang kèm bằng chứng lời thoại."
                    ),
                    user_prompt=prompt,
                    schema_cls=PrePassCastResult,
                    image_paths=imgs,
                )
                return res.characters
            except Exception:
                return []

        raw_chars: list[SeedCharacter] = []
        if max_workers > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
                for chunk_chars in pool.map(_scan_sample_batch, batches):
                    raw_chars.extend(chunk_chars)
        else:
            for batch in batches:
                raw_chars.extend(_scan_sample_batch(batch))

        # Keep grounded entries that have dialogue/text evidence
        grounded = [
            c
            for c in raw_chars
            if c.canonical_name.strip()
            and c.evidence_quote.strip()
            and not c.canonical_name.strip().startswith("[Chưa rõ")
        ]

        merged: list[SeedCharacter] = []
        if grounded and self.llm.has_text_credentials:
            try:
                merge_res = self.llm.complete_json(
                    agent_name="PrePassCastMerger",
                    system_prompt=(
                        "Hãy gộp các mục trùng tên (ví dụ tên gọi tắt và họ tên đầy đủ của cùng một nhân vật) "
                        "thành danh sách nhân vật chuẩn duy nhất (tối đa 12 nhân vật). "
                        "Lưu ý: giữ phân biệt giữa dạng người trẻ tuổi vs trưởng thành nếu ngoại hình khác hẳn "
                        "(hoặc ghi rõ trong appearance_vi). Giữ nguyên `best_panel_id` rõ mặt nhất và `evidence_quote` có thật."
                    ),
                    user_prompt=json.dumps([c.model_dump() for c in grounded], ensure_ascii=False),
                    schema_cls=PrePassCastResult,
                )
                merged = merge_res.characters
            except Exception:
                merged = grounded
        else:
            merged = grounded

        # Python deduplication safety net
        deduped: list[SeedCharacter] = []
        seen_keys: set[str] = set()
        for c in merged:
            key = c.canonical_name.strip().lower()
            if key and key not in seen_keys:
                seen_keys.add(key)
                deduped.append(c)

        repo_dir.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            PrePassCastResult(characters=deduped).model_dump_json(indent=2),
            encoding="utf-8",
        )
        built_sheet = _build_cast_sheet(deduped, manifest, sheet_path)
        return deduped, built_sheet

    @staticmethod
    def _format_seed_cast_hint(seed_cast: list[SeedCharacter], scene_state: dict) -> str:
        lines: list[str] = []
        if seed_cast:
            lines.append("### BẢNG NHÂN VẬT CHUẨN ĐÃ KHÓA (đối chiếu với cast_sheet.jpg nếu đính kèm):")
            for c in seed_cast[:12]:
                als = f" (còn gọi: {', '.join(c.aliases[:3])})" if c.aliases else ""
                lines.append(f"- {c.canonical_name}{als}: {c.appearance_vi}")
        if scene_state.get("recent_summaries"):
            lines.append("\n### NGỮ CẢNH CẢNH NGAY TRƯỚC ĐÓ (Rolling Scene State):")
            if scene_state.get("active_characters"):
                lines.append(
                    f"- Nhân vật đang có mặt trong cảnh: {', '.join(scene_state['active_characters'][:8])}"
                )
            lines.append("- Diễn biến 2-3 trang trước:")
            for s in scene_state["recent_summaries"][-3:]:
                lines.append(f"  * {s}")
        return "\n".join(lines) + ("\n\n" if lines else "")

    def analyze_page(
        self,
        page: PageInfo,
        known_cast: list[str] | None = None,
        seed_cast: list[SeedCharacter] | None = None,
        scene_state: dict | None = None,
        cast_sheet_path: Path | None = None,
    ) -> PageAnalysis:
        if _is_blank_or_solid_page(page.source_path):
            return _make_blank_page_analysis(page)

        panel_list = ", ".join(p.panel_id for p in page.panels)
        som_path = Path(page.som_paths[0]) if page.som_paths else Path(page.source_path)

        if self.llm.has_vision_credentials:
            hint = self._format_seed_cast_hint(seed_cast or [], scene_state or {})
            if not hint and known_cast:
                hint = f"Nhân vật đã xuất hiện: {', '.join(known_cast[:12])}.\n\n"

            imgs: list[Path] = []
            sheet_note = ""
            if cast_sheet_path and cast_sheet_path.exists():
                imgs.append(cast_sheet_path)
                sheet_note = "Ảnh ĐẦU TIÊN là BẢNG CHÂN DUNG `cast_sheet.jpg` để soi mặt/kiểu tóc.\n"
            imgs.append(som_path)

            user_prompt = (
                f"{sheet_note}"
                f"{hint}"
                f"Phân tích trang {page.page_id} (chương {page.chapter_id}, format={page.format}).\n"
                f"Các panel_id cần phân tích theo thứ tự: {panel_list}.\n"
                "Trả về JSON gồm page_id, chapter_id, page_summary (tiếng Việt), và danh sách panels "
                "(panel_id, action_summary, dialogue_summary, characters_present, emotion, importance 1-5)."
            )
            return self.llm.complete_json(
                agent_name="PageAnalyzerAgent",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema_cls=PageAnalysis,
                image_paths=imgs,
            )

        if not self.llm.config.offline_fallback:
            raise RuntimeError("No vision API credentials available for PageAnalyzerAgent.")

        panels_out: list[PanelAnalysis] = []
        for idx, p in enumerate(page.panels, start=1):
            has_face = any(fp.kind == "face" for fp in p.focal_points)
            importance = 4 if (has_face or p.height > p.width * 1.3) else 3
            emotion = "tense" if has_face else "neutral"
            panels_out.append(
                PanelAnalysis(
                    panel_id=p.panel_id,
                    action_summary=f"Khung {idx} trang {page.page_index} ({page.chapter_id}): diễn biến cảnh truyện.",
                    dialogue_summary="",
                    characters_present=[],
                    emotion=emotion,
                    importance=importance,
                )
            )
        return PageAnalysis(
            page_id=page.page_id,
            chapter_id=page.chapter_id,
            page_summary=f"Trang {page.page_index} của {page.chapter_id} gồm {len(page.panels)} khung hình.",
            panels=panels_out,
        )

    def analyze_batch(
        self,
        pages: list[PageInfo],
        known_cast: list[str] | None = None,
        seed_cast: list[SeedCharacter] | None = None,
        scene_state: dict | None = None,
        cast_sheet_path: Path | None = None,
        overlap_pages: list[PageInfo] | None = None,
    ) -> dict[str, PageAnalysis]:
        """Analyze consecutive pages in one Vision call with optional cast_sheet & overlap context."""
        blank_results: dict[str, PageAnalysis] = {}
        active_pages: list[PageInfo] = []
        for p in pages:
            if _is_blank_or_solid_page(p.source_path):
                blank_results[p.page_id] = _make_blank_page_analysis(p)
            else:
                active_pages.append(p)

        if not active_pages:
            return blank_results

        if len(active_pages) <= 1 or not self.llm.has_vision_credentials:
            for p in active_pages:
                blank_results[p.page_id] = self.analyze_page(
                    p,
                    known_cast=known_cast,
                    seed_cast=seed_cast,
                    scene_state=scene_state,
                    cast_sheet_path=cast_sheet_path,
                )
            return blank_results

        image_paths: list[Path] = []
        preamble_parts: list[str] = []
        img_cursor = 1

        if cast_sheet_path and cast_sheet_path.exists():
            image_paths.append(cast_sheet_path)
            preamble_parts.append(
                f"Ảnh #{img_cursor}: BẢNG CHÂN DUNG NHÂN VẬT (cast_sheet.jpg) — dùng để đối chiếu mặt & kiểu tóc."
            )
            img_cursor += 1

        if overlap_pages:
            ov_ids = []
            for op in overlap_pages:
                image_paths.append(Path(op.som_paths[0]) if op.som_paths else Path(op.source_path))
                ov_ids.append(f"Ảnh #{img_cursor}={op.page_id}")
                img_cursor += 1
            preamble_parts.append(
                f"Các trang gối đầu chỉ để hiểu mạch cảnh trước (KHÔNG xuất JSON cho các trang này): {', '.join(ov_ids)}."
            )

        page_specs = []
        for p in active_pages:
            image_paths.append(Path(p.som_paths[0]) if p.som_paths else Path(p.source_path))
            plist = ", ".join(pnl.panel_id for pnl in p.panels)
            page_specs.append(
                f"- Ảnh #{img_cursor}: page_id={p.page_id}, chapter_id={p.chapter_id}, panels=[{plist}]"
            )
            img_cursor += 1

        hint = self._format_seed_cast_hint(seed_cast or [], scene_state or {})
        if not hint and known_cast:
            hint = f"Nhân vật đã biết (giữ thống nhất tên): {', '.join(known_cast[:12])}.\n\n"

        preamble = ("\n".join(preamble_parts) + "\n\n") if preamble_parts else ""
        user_prompt = (
            preamble
            + hint
            + "Phân tích lần lượt từng trang manga cần xuất kết quả dưới đây bằng TIẾNG VIỆT:\n"
            + "\n".join(page_specs)
            + "\nTrả về JSON dạng PageBatchAnalysis gồm danh sách `pages`, mỗi phần tử có đúng `page_id`, `chapter_id`, `page_summary`, `panels`."
        )
        try:
            batch_res = self.llm.complete_json(
                agent_name="PageAnalyzerAgent",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema_cls=PageBatchAnalysis,
                image_paths=image_paths,
            )
            valid_ids = {p.page_id for p in active_pages}
            for pa in batch_res.pages:
                if pa.page_id in valid_ids:
                    blank_results[pa.page_id] = pa
            return blank_results
        except Exception:
            return blank_results

    def run_all(
        self,
        manifest: JobManifest,
        job_dir: Path,
        force: bool = False,
        max_workers: int = 4,
        batch_size: int = 4,
    ) -> list[PageAnalysis]:
        out_all = job_dir / "02a_all_pages.json"
        pages_dir = job_dir / "02a_pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        all_pages = [page for ch in manifest.chapters for page in ch.pages]
        page_idx_map = {pg.page_id: idx for idx, pg in enumerate(all_pages)}
        results: dict[str, PageAnalysis] = {}
        uncached: list[PageInfo] = []

        # Cache hygiene: reload valid cached pages and automatically re-queue any dummy placeholder pages
        for page in all_pages:
            page_json = pages_dir / f"{page.page_id}.json"
            if page_json.exists() and not force:
                try:
                    pa = PageAnalysis.model_validate_json(page_json.read_text(encoding="utf-8"))
                    if not _is_dummy_page_analysis(pa):
                        results[page.page_id] = pa
                        continue
                except Exception:
                    pass
            uncached.append(page)

        seed_cast, cast_sheet_path = self.discover_seed_cast(
            manifest, job_dir, force=force, max_workers=max_workers
        )
        known_cast_seed = [c.canonical_name for c in seed_cast]

        if uncached:
            step = max(1, batch_size)
            # Group uncached pages into contiguous sequential chunks for Chunked Parallel Rolling
            num_chunks = min(max(1, max_workers), max(1, len(uncached) // max(step, 8) + (1 if len(uncached) >= step else 0)))
            if max_workers <= 1 or len(uncached) <= step * 2:
                num_chunks = 1
            chunk_size = math.ceil(len(uncached) / num_chunks)
            worker_chunks = [
                uncached[i : i + chunk_size] for i in range(0, len(uncached), chunk_size)
            ]

            def _run_chunk(chunk_pages: list[PageInfo]) -> list[PageAnalysis]:
                # Initialize RollingSceneState from cached predecessors if available, else 2 overlap images on first batch
                state: dict = {
                    "active_characters": [],
                    "recent_summaries": [],
                }
                first_global_idx = page_idx_map.get(chunk_pages[0].page_id, 0)
                overlap_for_first: list[PageInfo] = []
                for prev_i in range(max(0, first_global_idx - 2), first_global_idx):
                    prev_pg = all_pages[prev_i]
                    if prev_pg.page_id in results:
                        prev_pa = results[prev_pg.page_id]
                        state["recent_summaries"].append(f"{prev_pa.page_id}: {prev_pa.page_summary}")
                        for pnl in prev_pa.panels:
                            for ch_name in pnl.characters_present:
                                if ch_name and ch_name not in state["active_characters"]:
                                    state["active_characters"].append(ch_name)
                    else:
                        overlap_for_first.append(prev_pg)

                chunk_out: list[PageAnalysis] = []
                batches = [chunk_pages[i : i + step] for i in range(0, len(chunk_pages), step)]
                for b_idx, batch in enumerate(batches):
                    got = self.analyze_batch(
                        batch,
                        known_cast=known_cast_seed or state["active_characters"] or None,
                        seed_cast=seed_cast,
                        scene_state=state,
                        cast_sheet_path=cast_sheet_path,
                        overlap_pages=(overlap_for_first if b_idx == 0 and overlap_for_first else None),
                    )
                    for pg in batch:
                        pa = got.get(pg.page_id)
                        if pa is None:
                            pa = self.analyze_page(
                                pg,
                                known_cast=known_cast_seed or state["active_characters"] or None,
                                seed_cast=seed_cast,
                                scene_state=state,
                                cast_sheet_path=cast_sheet_path,
                            )
                        # Only persist genuine Vision results (never poison disk cache with dummy placeholders)
                        if not _is_dummy_page_analysis(pa) or not self.llm.has_vision_credentials:
                            (pages_dir / f"{pg.page_id}.json").write_text(
                                pa.model_dump_json(indent=2), encoding="utf-8"
                            )

                        state["recent_summaries"].append(f"{pa.page_id}: {pa.page_summary}")
                        state["recent_summaries"] = state["recent_summaries"][-3:]
                        cur_chars: list[str] = []
                        for pnl in pa.panels:
                            for ch_name in pnl.characters_present:
                                clean_n = ch_name.strip()
                                if clean_n and clean_n not in cur_chars:
                                    cur_chars.append(clean_n)
                        if cur_chars:
                            state["active_characters"] = cur_chars[:8]

                        chunk_out.append(pa)
                return chunk_out

            if num_chunks > 1 and len(worker_chunks) > 1:
                with ThreadPoolExecutor(max_workers=num_chunks) as pool:
                    for chunk_out in pool.map(_run_chunk, worker_chunks):
                        for pa in chunk_out:
                            results[pa.page_id] = pa
            else:
                for chunk in worker_chunks:
                    for pa in _run_chunk(chunk):
                        results[pa.page_id] = pa

        analyses = [results[p.page_id] for p in all_pages]
        out_all.write_text(
            json.dumps([a.model_dump() for a in analyses], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return analyses
