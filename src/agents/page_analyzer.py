"""Stage 1.5 + 2a Agent: Pre-Pass Cast Discovery (cast_sheet.jpg) + Chunked Parallel Rolling Vision."""

from __future__ import annotations

import base64
import json
import math
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from src.llm.client import (
    LLMClient,
    extract_xml_attr,
    extract_xml_tag,
    iter_xml_blocks,
)
from src.schemas.analysis import PageAnalysis, PanelAnalysis
from src.schemas.ingestion import JobManifest, PageInfo, PanelInfo

if TYPE_CHECKING:
    from src.schemas.series import SeriesContext

_VERIFY_ALPHABET = "ACDEFGHJKMNPQRTUVWXY34679"


def _generate_verify_code(length: int = 4) -> str:
    """Generate a high-contrast OCR-unambiguous verification token."""
    return "".join(secrets.choice(_VERIFY_ALPHABET) for _ in range(length))

SYSTEM_PROMPT = """Bạn là PageAnalyzerAgent chuyên đọc trang manga/manhwa đã đánh dấu Set-of-Mark ([P1], [P2]...).
Quy tắc bắt buộc:
1) Viết 100% bằng TIẾNG VIỆT tự nhiên, chính xác (giữ nguyên tên riêng nhân vật, địa danh, tổ chức và thuật ngữ đặc trưng theo bản gốc).
2) Phân tích từng khung hình theo đúng thứ tự panel_id từ [P1] đến [Pn] theo chiều đọc.
3) Nếu đã đính kèm ảnh BẢNG CHÂN DUNG NHÂN VẬT (cast_sheet.jpg), BẮT BUỘC đối chiếu gương mặt, kiểu tóc và trang phục với bảng đó kết hợp với Ngữ cảnh cảnh trước (Rolling Scene State) để gọi ĐÚNG tên nhân vật, tránh nhầm lẫn nhân vật này sang nhân vật khác.
4) Trong `dialogue_summary`, tóm tắt hoặc trích dịch gọn lời thoại quan trọng nhất trong bóng thoại bằng tiếng Việt rõ nghĩa.
5) Trong `verbatim_quote_vi` (cấp trang), trích NGUYÊN VĂN câu thoại/câu dẫn đắt giá nhất trên trang dịch sang tiếng Việt tự nhiên (để rỗng "" nếu trang không có thoại đáng chú ý). Trong `visual_punch`, ghi ngắn gọn 1 chi tiết thị giác/biểu cảm đắt nhất trên trang (để rỗng "" nếu bình thường).
6) Nếu trang là trang bìa (cover), trang trắng/trang lót, trang tiêu đề (title page), mục lục (table of contents), lời cảm ơn (Special Thanks), chú thích dịch giả (Translator's Notes), trang bản quyền hay truyện ngắn tặng kèm cuối tập, BẮT BUỘC ghi rõ cụm "[PHỤ LỤC]" ở đầu `page_summary`.
7) CHỐNG BỊA TÊN (BẮT BUỘC): Chỉ dùng tên nhân vật có trong Bảng Chân Dung / Rolling Scene State hoặc được gọi rõ trong bóng thoại/chữ trên trang. TUYỆT ĐỐI KHÔNG tự đoán tên từ kiến thức ngoài truyện khi tên đó chưa hề xuất hiện; nếu là người lạ chưa xưng tên, hãy ghi dạng `[Chưa rõ tên: đặc điểm ngoại hình nổi bật]` (ví dụ `[Chưa rõ tên: người đeo mặt nạ mặc áo choàng tối màu]`).
8) CHỐNG BỊA CỐT TRUYỆN TRÊN TRANG BÌA / TRANG TRẮNG / MỤC LỤC (BẮT BUỘC): Nếu trang chỉ là trang trắng, trang bìa, trang tiêu đề hoặc mục lục (không có bóng thoại truyện), chỉ mô tả đúng những gì in trên trang (ví dụ: `[PHỤ LỤC] Trang trắng`, `[PHỤ LỤC] Trang mục lục tập 1`). TUYỆT ĐỐI KHÔNG lấy trí nhớ ngoài truyện hay các trang sau để bịa ra cảnh hành động hoặc lời thoại không có trên ảnh."""


class PageBatchAnalysis(BaseModel):
    pages: list[PageAnalysis] = Field(default_factory=list)


class SeedCharacter(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    appearance_vi: str = ""
    best_panel_id: str = ""
    portrait_path: str = ""


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


def _make_duplicate_spread_page_analysis(page: PageInfo) -> PageAnalysis:
    """Deterministic analysis for split pages duplicating an adjacent double-page spread."""
    return PageAnalysis(
        page_id=page.page_id,
        chapter_id=page.chapter_id,
        page_summary="[PHỤ LỤC] Trang đơn cắt trùng từ trang đôi (double-page spread).",
        panels=[
            PanelAnalysis(
                panel_id=p.panel_id,
                action_summary="[PHỤ LỤC] Trang đơn cắt trùng từ trang đôi.",
                dialogue_summary="",
                characters_present=[],
                emotion="neutral",
                importance=1,
            )
            for p in page.panels
        ],
    )


def _select_prepass_pages(
    manifest: JobManifest,
    max_samples: int = 16,
    exclude_pages: set[str] | None = None,
) -> list[PageInfo]:
    """Select face-rich, dialogue-rich pages distributed across the volume for Pre-Pass Cast Discovery."""
    all_pages = [pg for ch in manifest.chapters for pg in ch.pages]
    if len(all_pages) <= max_samples:
        return all_pages

    scored: list[tuple[float, int, PageInfo]] = []
    for idx, pg in enumerate(all_pages):
        if exclude_pages and pg.page_id in exclude_pages:
            continue
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


@lru_cache(maxsize=8)
def _load_cast_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    )
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def _draw_utf8_label(
    canvas_bgr: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_size: int = 14,
    color_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render UTF-8 Vietnamese text cleanly onto a BGR canvas using Pillow."""
    pil_img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=_load_cast_font(font_size), fill=color_rgb)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


_CAST_XML_TEMPLATE = """<cast>
  <character name="Tên chuẩn" best_panel_id="ch01_p001_P1">
    <aliases>Bí danh 1 | Bí danh 2</aliases>
    <evidence_quote>Câu thoại/chữ gốc gọi tên nhân vật</evidence_quote>
    <appearance>Mô tả ngắn ngoại hình</appearance>
  </character>
</cast>"""

_PAGE_XML_TEMPLATE = """<pages>
  <page id="ch01_p001" chapter="ch01" verify_code="XXXX">
    <summary>Tóm tắt diễn biến toàn trang bằng tiếng Việt (hoặc [PHỤ LỤC] ...)</summary>
    <verbatim_quote>Câu thoại/dẫn đắt giá nhất dịch sang tiếng Việt (để trống nếu không có)</verbatim_quote>
    <visual_punch>1 chi tiết thị giác/biểu cảm đắt nhất (để trống nếu bình thường)</visual_punch>
    <panels>
      <panel id="ch01_p001_P1" emotion="tense" importance="3">
        <characters>Tên 1 | Tên 2</characters>
        <action>Mô tả hành động/bố cục trong khung</action>
        <dialogue>Tóm tắt hoặc trích dịch lời thoại trong khung</dialogue>
      </panel>
    </panels>
  </page>
</pages>"""


def _split_char_field(raw: str) -> list[str]:
    if not raw.strip():
        return []
    parts = re.split(r"\s*\|\s*|,\s*(?![^\[]*\])", raw.strip())
    return [p.strip() for p in parts if p.strip()]


def _parse_cast_xml(xml_text: str) -> PrePassCastResult:
    chars: list[SeedCharacter] = []
    for attrs, body in iter_xml_blocks(xml_text, "character"):
        name = extract_xml_attr(attrs, "name") or extract_xml_tag(body, "canonical_name")
        best_pid = extract_xml_attr(attrs, "best_panel_id") or extract_xml_tag(body, "best_panel_id")
        aliases = _split_char_field(extract_xml_tag(body, "aliases"))
        ev = extract_xml_tag(body, "evidence_quote")
        app = extract_xml_tag(body, "appearance") or extract_xml_tag(body, "appearance_vi")
        if name:
            chars.append(
                SeedCharacter(
                    canonical_name=name,
                    aliases=aliases,
                    evidence_quote=ev,
                    appearance_vi=app,
                    best_panel_id=best_pid,
                )
            )
    return PrePassCastResult(characters=chars)


def _parse_pages_xml(
    xml_text: str,
    default_page_id: str = "",
    default_chapter_id: str = "",
) -> PageBatchAnalysis:
    pages: list[PageAnalysis] = []
    blocks = iter_xml_blocks(xml_text, "page")
    if not blocks and default_page_id and "<panels" in xml_text.lower():
        blocks = [(f'id="{default_page_id}" chapter="{default_chapter_id}"', xml_text)]
    for attrs, body in blocks:
        pid = extract_xml_attr(attrs, "id") or extract_xml_tag(body, "page_id") or default_page_id
        cid = extract_xml_attr(attrs, "chapter") or extract_xml_tag(body, "chapter_id") or default_chapter_id
        if not cid and "_" in pid:
            cid = pid.split("_", 1)[0]
        vcode = extract_xml_attr(attrs, "verify_code") or extract_xml_tag(body, "verify_code")
        summary = extract_xml_tag(body, "summary") or extract_xml_tag(body, "page_summary")
        vquote = extract_xml_tag(body, "verbatim_quote") or extract_xml_tag(body, "verbatim_quote_vi")
        vpunch = extract_xml_tag(body, "visual_punch")
        pnls: list[PanelAnalysis] = []
        for p_attrs, p_body in iter_xml_blocks(body, "panel"):
            pnl_id = extract_xml_attr(p_attrs, "id") or extract_xml_tag(p_body, "panel_id")
            if not pnl_id:
                continue
            emo = extract_xml_attr(p_attrs, "emotion") or extract_xml_tag(p_body, "emotion") or "neutral"
            imp_raw = extract_xml_attr(p_attrs, "importance") or extract_xml_tag(p_body, "importance") or "3"
            try:
                imp = max(1, min(5, int(re.findall(r"\d+", imp_raw)[0])))
            except Exception:
                imp = 3
            chars = _split_char_field(extract_xml_tag(p_body, "characters"))
            act = extract_xml_tag(p_body, "action") or extract_xml_tag(p_body, "action_summary") or ""
            dlg = extract_xml_tag(p_body, "dialogue") or extract_xml_tag(p_body, "dialogue_summary") or ""
            pnls.append(
                PanelAnalysis(
                    panel_id=pnl_id,
                    action_summary=act,
                    dialogue_summary=dlg,
                    characters_present=chars,
                    emotion=emo,
                    importance=imp,
                )
            )
        if pid:
            pages.append(
                PageAnalysis(
                    page_id=pid,
                    chapter_id=cid or "ch01",
                    visual_verify_code=vcode.strip().upper(),
                    page_summary=summary,
                    verbatim_quote_vi=vquote,
                    visual_punch=vpunch,
                    panels=pnls,
                )
            )
    return PageBatchAnalysis(pages=pages)


def _stamp_verify_banner_data_url(
    image_path: Path,
    page_id: str,
    code: str,
) -> tuple[Path | str, bool]:
    """Prepend a 40px high-contrast header banner with random OCR code in RAM (never modifies disk)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return image_path, False
    banner_h = 40
    bordered = cv2.copyMakeBorder(img, banner_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(16, 16, 20))
    label = f"[ PAGE: {page_id}  |  CODE: {code} ]"
    bordered = _draw_utf8_label(bordered, label, (12, 8), font_size=22, color_rgb=(255, 232, 80))
    ok, buf = cv2.imencode(".jpg", bordered, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return image_path, False
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", True


def _build_cast_sheet(
    seed_cast: list[SeedCharacter],
    manifest: JobManifest,
    out_path: Path,
    max_cards: int = 16,
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
        portrait: np.ndarray | None = None
        # Prefer persistent Portrait Vault tile when inherited from Series Memory
        if ch_entry.portrait_path and Path(ch_entry.portrait_path).exists():
            loaded = cv2.imread(ch_entry.portrait_path)
            if loaded is not None and loaded.size > 0:
                if loaded.shape[0] != tile_h or loaded.shape[1] != tile_w:
                    loaded = cv2.resize(loaded, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                portrait = loaded

        if portrait is None:
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
        card = _draw_utf8_label(card, label, (8, card_h - 27), font_size=14, color_rgb=(255, 255, 255))
        cards.append(card)

    if not cards:
        return None

    cols = min(4, max(2, math.ceil(math.sqrt(len(cards)))))
    rows = math.ceil(len(cards) / cols)
    pad = 8
    grid_w = cols * card_w + (cols + 1) * pad
    grid_h = rows * card_h + (rows + 1) * pad + 28
    canvas = np.full((grid_h, grid_w, 3), 10, dtype=np.uint8)
    canvas = _draw_utf8_label(
        canvas,
        "CAST REFERENCE SHEET (match faces & hairstyles strictly)",
        (pad, 6),
        font_size=15,
        color_rgb=(255, 230, 120),
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
        exclude_pages: set[str] | None = None,
        prev_job_dir: Path | None = None,
        series_ctx: SeriesContext | None = None,
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
            if series_ctx is not None and series_ctx.state.characters:
                from src.series.memory import SeriesStore

                inherited = SeriesStore.select_active_cast_for_volume(
                    series_ctx.state,
                    series_ctx.volume_index,
                    local_seed_cast=[],
                    max_cards=16,
                )
                if inherited:
                    repo_dir.mkdir(parents=True, exist_ok=True)
                    seed_path.write_text(
                        PrePassCastResult(characters=inherited).model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    built_sheet = _build_cast_sheet(inherited, manifest, sheet_path, max_cards=16)
                    return inherited, built_sheet
            return [], None

        samples = _select_prepass_pages(manifest, max_samples=16, exclude_pages=exclude_pages)
        batches = [samples[i : i + 4] for i in range(0, len(samples), 4)]
        series_cast_hint = ""
        if series_ctx is not None and series_ctx.state.characters:
            known_lines = [
                f"- {ch.canonical_name} ({ch.appearance[:80]})"
                for ch in series_ctx.state.characters[:20]
                if ch.status != "deceased"
            ]
            if known_lines:
                series_cast_hint = (
                    "\nNHÂN VẬT ĐÃ BIẾT TỪ CÁC TẬP TRƯỚC (ưu tiên giữ đúng tên chuẩn này nếu nhận ra gương mặt):\n"
                    + "\n".join(known_lines)
                    + "\n"
                )

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
                "3) `best_panel_id`: chọn đúng 1 panel_id nơi nhân vật lộ mặt to, rõ nhất (ưu tiên panel chỉ có 1 người).\n"
                f"{series_cast_hint}\n"
                f"Các trang đính kèm:\n{specs}\n"
                "Trả về XML `<cast>` chứa các thẻ `<character>` (name, best_panel_id, aliases, evidence_quote, appearance)."
            )
            try:
                res = self.llm.complete_xml(
                    agent_name="PrePassCastScanner",
                    system_prompt=(
                        "Bạn là chuyên gia nhận diện nhân vật manga đen trắng. "
                        "Chỉ trích xuất tên nhân vật thực sự xuất hiện trên trang kèm bằng chứng lời thoại."
                    ),
                    user_prompt=prompt,
                    xml_template=_CAST_XML_TEMPLATE,
                    parse_fn=_parse_cast_xml,
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
                merge_res = self.llm.complete_xml(
                    agent_name="PrePassCastMerger",
                    system_prompt=(
                        "Hãy gộp các mục trùng tên (ví dụ tên gọi tắt và họ tên đầy đủ của cùng một nhân vật) "
                        "thành danh sách nhân vật chuẩn duy nhất (tối đa 12 nhân vật). "
                        "Lưu ý: giữ phân biệt giữa dạng người trẻ tuổi vs trưởng thành nếu ngoại hình khác hẳn "
                        "(hoặc ghi rõ trong appearance). Giữ nguyên `best_panel_id` rõ mặt nhất và `evidence_quote` có thật."
                    ),
                    user_prompt=json.dumps([c.model_dump() for c in grounded], ensure_ascii=False),
                    xml_template=_CAST_XML_TEMPLATE,
                    parse_fn=_parse_cast_xml,
                )
                merged = merge_res.characters
            except Exception:
                merged = grounded
        else:
            merged = grounded

        # Python deduplication safety net (Series Portrait Vault takes precedence over legacy --prev-job)
        deduped: list[SeedCharacter] = []
        if series_ctx is not None:
            from src.series.memory import SeriesStore

            deduped = SeriesStore.select_active_cast_for_volume(
                series_ctx.state,
                series_ctx.volume_index,
                local_seed_cast=merged,
                max_cards=16,
            )
        else:
            seen_keys: set[str] = set()
            if prev_job_dir is not None:
                prev_seed_path = prev_job_dir / "repo" / "seed_cast.json"
                if prev_seed_path.exists():
                    try:
                        prev_res = PrePassCastResult.model_validate_json(
                            prev_seed_path.read_text(encoding="utf-8")
                        )
                        for pc in prev_res.characters:
                            pk = pc.canonical_name.strip().lower()
                            if pk and pk not in seen_keys:
                                seen_keys.add(pk)
                                # Clear stale panel_id from prior volume so we never mis-crop
                                pc_copy = pc.model_copy(deep=True)
                                pc_copy.best_panel_id = ""
                                deduped.append(pc_copy)
                    except Exception:
                        pass
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
        built_sheet = _build_cast_sheet(deduped, manifest, sheet_path, max_cards=16)
        return deduped, built_sheet

    @staticmethod
    def _format_seed_cast_hint(seed_cast: list[SeedCharacter], scene_state: dict) -> str:
        lines: list[str] = []
        if seed_cast:
            lines.append("### BẢNG NHÂN VẬT CHUẨN ĐÃ KHÓA (đối chiếu với cast_sheet.jpg nếu đính kèm):")
            for c in seed_cast[:16]:
                als = f" (còn gọi: {', '.join(c.aliases[:3])})" if c.aliases else ""
                lines.append(f"- {c.canonical_name}{als}: {c.appearance_vi}")
        if scene_state.get("glossary_terms"):
            terms = "; ".join(f"{k} -> {v}" for k, v in list(scene_state["glossary_terms"].items())[:20])
            lines.append(f"\n### TỪ ĐIỂN THUẬT NGỮ CHUẨN CỦA SERIES (dịch nhất quán): {terms}")
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

            imgs: list[Path | str] = []
            sheet_note = ""
            if cast_sheet_path and cast_sheet_path.exists():
                imgs.append(cast_sheet_path)
                sheet_note = "Ảnh ĐẦU TIÊN là BẢNG CHÂN DUNG `cast_sheet.jpg` để soi mặt/kiểu tóc.\n"
            expected_code = _generate_verify_code()
            stamped_img, was_stamped = _stamp_verify_banner_data_url(som_path, page.page_id, expected_code)
            imgs.append(stamped_img)

            user_prompt = (
                f"{sheet_note}"
                f"{hint}"
                f"Phân tích trang {page.page_id} (chương {page.chapter_id}, format={page.format}).\n"
                f"Các panel_id cần phân tích theo thứ tự: {panel_list}.\n"
                "BẮT BUỘC đọc đúng mã 4 ký tự in trên thanh tiêu đề vàng trên cùng của ảnh (`CODE: XXXX`) và điền vào `verify_code`.\n"
                "Trả về XML `<pages>` chứa 1 thẻ `<page>` với đầy đủ `verify_code`, `<summary>`, `<verbatim_quote>`, `<visual_punch>`, và các `<panel>`."
            )
            batch_res = self.llm.complete_xml(
                agent_name="PageAnalyzerAgent",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                xml_template=_PAGE_XML_TEMPLATE,
                parse_fn=lambda txt: _parse_pages_xml(
                    txt, default_page_id=page.page_id, default_chapter_id=page.chapter_id
                ),
                image_paths=imgs,
            )
            pa = next((x for x in batch_res.pages if x.page_id == page.page_id), None)
            if pa is None and batch_res.pages:
                pa = batch_res.pages[0]
            if pa is None:
                raise RuntimeError(f"Empty XML <page> response for {page.page_id}")
            if was_stamped and pa.visual_verify_code.strip().upper() != expected_code:
                msg = (
                    f"Visual verify code mismatch on {page.page_id}: "
                    f"got {pa.visual_verify_code!r}, expected {expected_code!r} (image may have been dropped)."
                )
                if self.llm.telemetry:
                    self.llm.telemetry.warn(msg)
                raise RuntimeError(msg)
            return pa

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

        image_paths: list[Path | str] = []
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
                f"Các trang gối đầu chỉ để hiểu mạch cảnh trước (KHÔNG xuất XML cho các trang này): {', '.join(ov_ids)}."
            )

        page_specs = []
        expected_codes: dict[str, str] = {}
        stamped_flags: dict[str, bool] = {}
        for p in active_pages:
            src_img = Path(p.som_paths[0]) if p.som_paths else Path(p.source_path)
            code = _generate_verify_code()
            expected_codes[p.page_id] = code
            stamped_img, was_stamped = _stamp_verify_banner_data_url(src_img, p.page_id, code)
            stamped_flags[p.page_id] = was_stamped
            image_paths.append(stamped_img)
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
            + "Phân tích lần lượt từng trang manga cần xuất kết quả dưới đây bằng TIẾNG VIỆT.\n"
            + "BẮT BUỘC với MỖI ảnh trang truyện, đọc mã 4 ký tự in trên thanh tiêu đề vàng trên cùng (`CODE: XXXX`) và ghi đúng vào `verify_code`:\n"
            + "\n".join(page_specs)
            + "\nTrả về XML dạng `<pages>` chứa danh sách `<page id=... chapter=... verify_code=...>` với đầy đủ `<summary>`, `<verbatim_quote>`, `<visual_punch>`, và các `<panel>`."
        )
        try:
            batch_res = self.llm.complete_xml(
                agent_name="PageAnalyzerAgent",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                xml_template=_PAGE_XML_TEMPLATE,
                parse_fn=_parse_pages_xml,
                image_paths=image_paths,
            )
            valid_ids = {p.page_id for p in active_pages}
            for pa in batch_res.pages:
                if pa.page_id not in valid_ids:
                    continue
                if stamped_flags.get(pa.page_id) and pa.visual_verify_code.strip().upper() != expected_codes[pa.page_id]:
                    if self.llm.telemetry:
                        self.llm.telemetry.warn(
                            f"PageAnalyzerAgent: batch verify_code mismatch on {pa.page_id} "
                            f"(got {pa.visual_verify_code!r}, expected {expected_codes[pa.page_id]!r}); forcing single-page retry."
                        )
                    continue
                blank_results[pa.page_id] = pa
            return blank_results
        except Exception:
            return blank_results

    @staticmethod
    def _build_adaptive_batches(
        pages: list[PageInfo],
        batch_size: int = 4,
        target_panels: int = 16,
    ) -> list[list[PageInfo]]:
        """Group consecutive pages adaptively by panel density (~16-18 panels/call)."""
        if batch_size <= 2 or len(pages) <= 2:
            step = max(1, batch_size)
            return [pages[i : i + step] for i in range(0, len(pages), step)]

        min_pages = 2
        max_pages = max(batch_size, min(6, batch_size + 1))
        batches: list[list[PageInfo]] = []
        cur_batch: list[PageInfo] = []
        cur_panels = 0

        for pg in pages:
            p_cnt = max(1, len(pg.panels))
            if cur_batch and (
                len(cur_batch) >= max_pages
                or (len(cur_batch) >= min_pages and cur_panels + p_cnt > target_panels + 2)
            ):
                batches.append(cur_batch)
                cur_batch = [pg]
                cur_panels = p_cnt
            else:
                cur_batch.append(pg)
                cur_panels += p_cnt

        if cur_batch:
            batches.append(cur_batch)
        return batches

    def run_all(
        self,
        manifest: JobManifest,
        job_dir: Path,
        force: bool = False,
        max_workers: int = 4,
        batch_size: int = 4,
        prev_job_dir: Path | None = None,
        series_ctx: SeriesContext | None = None,
    ) -> list[PageAnalysis]:
        out_all = job_dir / "02a_all_pages.json"
        pages_dir = job_dir / "02a_pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        from src.agents.page_index import _detect_duplicate_spread_pages

        dup_spread_pages = _detect_duplicate_spread_pages(manifest)
        all_pages = [page for ch in manifest.chapters for page in ch.pages]
        page_idx_map = {pg.page_id: idx for idx, pg in enumerate(all_pages)}
        results: dict[str, PageAnalysis] = {}
        uncached: list[PageInfo] = []

        # Cache hygiene: skip spread-duplicate splits deterministically before calling Vision LLM
        for page in all_pages:
            page_json = pages_dir / f"{page.page_id}.json"
            if page.page_id in dup_spread_pages:
                dup_pa = _make_duplicate_spread_page_analysis(page)
                results[page.page_id] = dup_pa
                page_json.write_text(dup_pa.model_dump_json(indent=2), encoding="utf-8")
                continue
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
            manifest,
            job_dir,
            force=force,
            max_workers=max_workers,
            exclude_pages=dup_spread_pages,
            prev_job_dir=prev_job_dir,
            series_ctx=series_ctx,
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
                # Initialize RollingSceneState from cached predecessors (or Series handoff on page 0)
                state: dict = {
                    "active_characters": [],
                    "recent_summaries": [],
                    "glossary_terms": (
                        dict(series_ctx.glossary.terminology)
                        if series_ctx is not None and series_ctx.glossary.terminology
                        else {}
                    ),
                }
                first_global_idx = page_idx_map.get(chunk_pages[0].page_id, 0)
                if (
                    first_global_idx == 0
                    and series_ctx is not None
                    and series_ctx.state.last_handoff is not None
                    and series_ctx.state.last_handoff.volume_index < series_ctx.volume_index
                ):
                    ho = series_ctx.state.last_handoff
                    if ho.ending_characters:
                        state["active_characters"] = list(ho.ending_characters[:8])
                    if ho.ending_summary:
                        state["recent_summaries"].append(f"[Cuối tập {ho.volume_index}] {ho.ending_summary}")
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
                batches = self._build_adaptive_batches(chunk_pages, batch_size=step)
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
                            last_exc: Exception | None = None
                            for _ in range(2):
                                try:
                                    pa = self.analyze_page(
                                        pg,
                                        known_cast=known_cast_seed or state["active_characters"] or None,
                                        seed_cast=seed_cast,
                                        scene_state=state,
                                        cast_sheet_path=cast_sheet_path,
                                    )
                                    break
                                except Exception as exc:
                                    last_exc = exc
                            if pa is None:
                                raise RuntimeError(
                                    f"PageAnalyzerAgent failed visual verification after retries on {pg.page_id}: {last_exc}"
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
