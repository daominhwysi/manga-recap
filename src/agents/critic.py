"""Stage 3b Agent (Critic V3): Deterministic Gate + Repetition Scanner + Read-Only Semantic Auditor + Surgical Repairer."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.agents.alignment import (
    build_alignment_anchors,
    page_analyses_to_alignment_records,
    repair_part_panel_alignment_deterministic,
    verify_beats_alignment,
)
from src.agents.page_index import extract_distinctive_char_patterns
from src.audio.base import apply_pronunciation_glossary
from src.config import load_personas
from src.llm.client import (
    LLMClient,
    extract_xml_attr,
    extract_xml_tag,
    iter_xml_blocks,
)
from src.schemas.analysis import PageAnalysis, StoryBible
from src.schemas.ingestion import JobManifest
from src.schemas.script import CriticIssue, CriticReport, RecapScript, ScriptBeat

if TYPE_CHECKING:
    from src.config import ScriptConfig


_BANNED_CLICHES = (
    "định mệnh nghiệt ngã",
    "bánh răng định mệnh",
    "không gian vỡ vụn",
    "xé toạc không gian",
    "rợn tóc gáy",
    "lưỡi dao cào xé",
    "bàn cờ chính trị đẫm máu",
    "chuyển sang đầu phần mới",
    "chuyển sang phần mới",
    "nối tiếp cao trào",
    "bước sang diễn biến tiếp theo",
    "sang hồi kết mở",
)

# Common Vietnamese grammatical tokens that are not meaningful standalone content words
_VN_STOPWORDS = {
    "và",
    "của",
    "là",
    "có",
    "cho",
    "với",
    "từ",
    "trong",
    "ngoài",
    "trên",
    "dưới",
    "về",
    "tới",
    "đến",
    "vào",
    "ra",
    "lên",
    "xuống",
    "đi",
    "lại",
    "đã",
    "đang",
    "sẽ",
    "vừa",
    "mới",
    "còn",
    "cũng",
    "chỉ",
    "thì",
    "mà",
    "nhưng",
    "rồi",
    "khi",
    "lúc",
    "này",
    "kia",
    "đó",
    "ấy",
    "một",
    "hai",
    "các",
    "những",
    "cậu",
    "cô",
    "anh",
    "ông",
    "bà",
    "chú",
    "hắn",
    "họ",
    "mình",
    "người",
    "bị",
    "được",
    "không",
    "chẳng",
    "chưa",
    "rất",
    "quá",
    "hơn",
    "nhất",
}

_TEXT_FIXABLE_CATEGORIES = {
    "spoiler_name",
    "duplication",
    "style",
    "budget",
    "pacing",
    "opener_repeat",
    "tic_repeat",
    "seam_echo",
    "semantic_repeat",
    "series_continuity",
}

# General Vietnamese conversational filler tics/particles (capped globally across any position in beat)
_GLOBAL_DISCOURSE_TICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nha", re.compile(r"(?<!\w)nha(?:[,.:;!?…—–]|\s|$)", flags=re.IGNORECASE | re.UNICODE)),
    ("nhé", re.compile(r"(?<!\w)nhé(?:[,.:;!?…—–]|\s|$)", flags=re.IGNORECASE | re.UNICODE)),
    ("kìa", re.compile(r"(?<!\w)kìa(?:[,.:;!?…—–]|\s|$)", flags=re.IGNORECASE | re.UNICODE)),
    ("để ý", re.compile(r"(?<!\w)để ý(?!\w)", flags=re.IGNORECASE | re.UNICODE)),
    ("nghe kỹ", re.compile(r"(?<!\w)nghe kỹ(?!\w)", flags=re.IGNORECASE | re.UNICODE)),
    ("anh em", re.compile(r"(?<!\w)anh em(?!\w)", flags=re.IGNORECASE | re.UNICODE)),
    ("khoan", re.compile(r"(?<!\w)khoan(?:[,.:;!?…—–]|\s|$)", flags=re.IGNORECASE | re.UNICODE)),
    ("ai ngờ", re.compile(r"(?<!\w)ai ngờ(?!\w)", flags=re.IGNORECASE | re.UNICODE)),
    ("cứ tưởng", re.compile(r"(?<!\w)cứ tưởng(?!\w)", flags=re.IGNORECASE | re.UNICODE)),
    ("tới lúc này", re.compile(r"(?<!\w)tới lúc này(?!\w)", flags=re.IGNORECASE | re.UNICODE)),
)


def _strip_quoted_dialogue(text: str) -> str:
    """Remove direct quoted character dialogue so character speech isn't penalized as narrator tics."""
    return re.sub(r'["“][^"”]*["”]', " ", text)


def _page_pos_from_panel(panel_id: str, page_order: dict[str, int]) -> int | None:
    pg = re.sub(r"_P\d+$", "", panel_id)
    return page_order.get(pg)


def _content_ngrams(text: str) -> set[str]:
    """Extract normalized word 2-grams and 3-grams to catch cross-beat content repetition."""
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    ngrams: set[str] = set()
    for n in (2, 3):
        for i in range(max(0, len(words) - n + 1)):
            ngrams.add(" ".join(words[i : i + n]))
    return ngrams


def _extract_bible_name_tokens(bible: StoryBible | None) -> set[str]:
    """Collect lowercase character name tokens so normal subject mentions aren't mistaken for catchphrases."""
    out: set[str] = set()
    if not bible or not bible.characters:
        return out
    for ch in bible.characters:
        for raw in [ch.canonical_name, *ch.aliases]:
            for tok in re.findall(r"[^\W\d_]+", raw.lower(), flags=re.UNICODE):
                if len(tok) >= 2:
                    out.add(tok)
    return out


def extract_beat_openers(text: str, name_tokens: set[str] | None = None) -> list[str]:
    """Extract candidate conversational openers/fillers from sentence starts (0 hardcoding, purely structural)."""
    names = name_tokens or set()
    candidates: list[str] = []
    # Split into sentences to catch mid-beat repeated fillers like 'Nhưng để ý nha,' or 'Khoan,'
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]
    for s_idx, sent in enumerate(sentences):
        # Skip direct quoted dialogue
        if sent.startswith(('"', "“", "'")):
            continue
        # 1) Short clause (1-3 words) right before comma/colon/exclamation/dash at sentence start
        m_lead = re.match(r"^([^,.:;!?—–\"“”]{3,28})[,:!—–]", sent)
        if m_lead:
            lead_words = re.findall(r"[^\W\d_]+", m_lead.group(1).lower(), flags=re.UNICODE)
            if 1 <= len(lead_words) <= 3:
                # Strip single leading conjunction like 'nhưng' / 'còn' so 'nhưng khoan' matches 'khoan'
                trimmed = lead_words[1:] if len(lead_words) >= 2 and lead_words[0] in ("nhưng", "còn", "mà", "và") else lead_words
                if trimmed and trimmed[0] not in names:
                    phrase = " ".join(trimmed)
                    if (len(trimmed) == 1 and len(phrase) >= 5 and phrase not in _VN_STOPWORDS) or (
                        len(trimmed) >= 2 and any(w not in _VN_STOPWORDS for w in trimmed)
                    ):
                        candidates.append(phrase)

        # 2) Leading 2-3 word beat opener on the opening sentence only
        if s_idx == 0:
            words = re.findall(r"[^\W\d_]+", sent.lower(), flags=re.UNICODE)
            if len(words) >= 2 and words[0] not in names and words[1] not in names:
                for n in (3, 2):
                    if len(words) >= n:
                        sub = words[:n]
                        if any(w not in _VN_STOPWORDS for w in sub):
                            phrase = " ".join(sub)
                            if len(phrase) >= 6:
                                candidates.append(phrase)
    # Deduplicate preserving order
    return list(dict.fromkeys(candidates))


def _tail_head_content_words(text: str, is_tail: bool, name_tokens: set[str]) -> list[str]:
    """Return meaningful non-name content words from the last or first sentence of a beat."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]
    if not sents:
        return []
    target = sents[-1] if is_tail else sents[0]
    words = re.findall(r"[^\W\d_]+", target.lower(), flags=re.UNICODE)
    return [w for w in words if len(w) >= 3 and w not in _VN_STOPWORDS and w not in name_tokens]


class _SeamRewrite(BaseModel):
    beat_id: str
    narration_vi: str


class _SeamPolishResponse(BaseModel):
    rewrites: list[_SeamRewrite] = Field(default_factory=list)


class _SemanticAuditFlag(BaseModel):
    beat_id: str
    reason: str


class _SemanticAuditResponse(BaseModel):
    flags: list[_SemanticAuditFlag] = Field(default_factory=list)


class _PanelRepairItem(BaseModel):
    beat_id: str
    panel_ids: list[str] = Field(default_factory=list)


class _PanelRepairResponse(BaseModel):
    repairs: list[_PanelRepairItem] = Field(default_factory=list)


_SEAM_POLISH_XML_TEMPLATE = """<rewrites>
  <rewrite beat_id="b001">Câu kể văn nói tiếng Việt đã được sửa đúng lỗi...</rewrite>
</rewrites>"""

_SEMANTIC_AUDIT_XML_TEMPLATE = """<audit>
  <flag beat_id="b021" reason="Lặp lại hành động vừa kết ở b020"/>
</audit>"""

_PANEL_REPAIR_XML_TEMPLATE = """<panel_repairs>
  <repair beat_id="b015">
    <panels>ch01_p030_P3, ch01_p030_P4</panels>
  </repair>
</panel_repairs>"""


def _parse_seam_polish_xml(xml_text: str) -> _SeamPolishResponse:
    rewrites: list[_SeamRewrite] = []
    for attrs, body in iter_xml_blocks(xml_text, "rewrite"):
        bid = extract_xml_attr(attrs, "beat_id") or extract_xml_attr(attrs, "id") or extract_xml_tag(body, "beat_id")
        txt = extract_xml_tag(body, "narration") or re.sub(r"<[^>]+>", "", body).strip()
        if bid and txt:
            rewrites.append(_SeamRewrite(beat_id=bid, narration_vi=txt))
    return _SeamPolishResponse(rewrites=rewrites)


def _parse_semantic_audit_xml(xml_text: str) -> _SemanticAuditResponse:
    flags: list[_SemanticAuditFlag] = []
    # Support both self-closing <flag beat_id="..." reason="..."/> and <flag ...>reason</flag>
    for m in re.finditer(r"<flag\s+([^>]+?)/?>", xml_text, flags=re.IGNORECASE | re.DOTALL):
        attrs = m.group(1)
        bid = extract_xml_attr(attrs, "beat_id") or extract_xml_attr(attrs, "id")
        reason = extract_xml_attr(attrs, "reason")
        if bid and reason:
            flags.append(_SemanticAuditFlag(beat_id=bid, reason=reason))
    if not flags:
        for attrs, body in iter_xml_blocks(xml_text, "flag"):
            bid = extract_xml_attr(attrs, "beat_id") or extract_xml_attr(attrs, "id") or extract_xml_tag(body, "beat_id")
            reason = extract_xml_attr(attrs, "reason") or extract_xml_tag(body, "reason") or re.sub(r"<[^>]+>", "", body).strip()
            if bid and reason:
                flags.append(_SemanticAuditFlag(beat_id=bid, reason=reason))
    return _SemanticAuditResponse(flags=flags)


def _parse_panel_repair_xml(xml_text: str) -> _PanelRepairResponse:
    repairs: list[_PanelRepairItem] = []
    for attrs, body in iter_xml_blocks(xml_text, "repair"):
        bid = extract_xml_attr(attrs, "beat_id") or extract_xml_attr(attrs, "id") or extract_xml_tag(body, "beat_id")
        raw_panels = extract_xml_tag(body, "panels") or extract_xml_tag(body, "panel_ids") or re.sub(r"<[^>]+>", "", body)
        pids = [p.strip("[]() \"',.") for p in re.split(r"[\s,;]+", raw_panels.strip()) if p.strip("[]() \"',.")]
        if bid and pids:
            repairs.append(_PanelRepairItem(beat_id=bid, panel_ids=pids))
    return _PanelRepairResponse(repairs=repairs)


def _has_dialogue_quote(text: str) -> bool:
    return bool(re.search(r'["“][^"”]{3,}["”]', text))


class CriticAgent:
    """Critic V3: Deterministic Gate + Repetition Scanner + Read-Only Auditor + Allowlisted Surgical Repairer."""

    def __init__(self, llm: LLMClient, polish_seams: bool = True) -> None:
        self.llm = llm
        self.polish_seams = polish_seams

    @staticmethod
    def _scoped_fallback_panel(
        beat_idx: int,
        total_beats: int,
        valid_ids: list[str],
        cleaned_beats: list[ScriptBeat],
    ) -> str:
        if cleaned_beats and cleaned_beats[-1].panel_ids:
            last_pid = cleaned_beats[-1].panel_ids[-1]
            if last_pid in valid_ids:
                pos = valid_ids.index(last_pid)
                return valid_ids[min(len(valid_ids) - 1, pos + 1)]
        ratio = beat_idx / max(1, total_beats)
        return valid_ids[min(len(valid_ids) - 1, int(ratio * len(valid_ids)))]

    @staticmethod
    def _build_reveal_checks(
        bible: StoryBible | None,
        page_order: dict[str, int],
    ) -> list[tuple[str, str, int, list[re.Pattern[str]]]]:
        reveal_checks: list[tuple[str, str, int, list[re.Pattern[str]]]] = []
        if bible and bible.characters:
            canon_set = {c.canonical_name.strip().lower() for c in bible.characters if c.canonical_name.strip()}
            for ch in bible.characters:
                if ch.first_named_page_id and ch.first_named_page_id in page_order:
                    pats = extract_distinctive_char_patterns(ch, canon_set)
                    if pats:
                        reveal_checks.append(
                            (ch.canonical_name, ch.first_named_page_id, page_order[ch.first_named_page_id], pats)
                        )
        return reveal_checks

    @staticmethod
    def _evaluate_issues(
        beats: list[ScriptBeat],
        all_pages: list[str],
        page_order: dict[str, int],
        beat_to_allowed: dict[int, set[str]],
        target_total_words: int | None = None,
        bible: StoryBible | None = None,
        page_analyses: list[PageAnalysis] | None = None,
        part_bounds: list[dict] | None = None,
        series_ctx=None,
    ) -> tuple[list[CriticIssue], set[int], list[str]]:
        """Layer 1: Evaluate hard invariants + semantic panel alignment + statistical opener/seam repetition."""
        issues: list[CriticIssue] = []
        flagged_indices: set[int] = set()
        flagged_notes: list[str] = []
        total_beats = len(beats)
        avg_pages_per_beat = (len(all_pages) / max(1, total_beats)) if total_beats >= 6 else 2.5
        max_single_beat_span = max(4, math.ceil(avg_pages_per_beat * 2.0))
        max_coverage_gap = max(8, math.ceil(avg_pages_per_beat * 3.5))

        reveal_checks = CriticAgent._build_reveal_checks(bible, page_order)
        name_tokens = _extract_bible_name_tokens(bible)

        prev_page_pos: int | None = None
        beat_ngrams: list[tuple[str, set[str]]] = []
        word_counts: list[int] = []
        opener_hits: dict[str, list[int]] = {}
        tic_hits: dict[str, list[int]] = {}
        colon_beats = 0
        staccato_beats = 0

        for idx, beat in enumerate(beats):
            kept = beat.panel_ids
            if idx in beat_to_allowed and kept:
                allowed = beat_to_allowed[idx]
                out_ids = [pid for pid in kept if pid not in allowed]
                if out_ids:
                    msg = f"{beat.beat_id} references panels outside assigned part range: {out_ids}."
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="panel_sync", message=msg)
                    )
                    flagged_notes.append(msg)

            positions = [p for pid in kept if (p := _page_pos_from_panel(pid, page_order)) is not None]
            if positions and beat.beat_type != "hook_montage":
                cur_min = min(positions)
                cur_max = max(positions)
                if cur_max - cur_min > max_single_beat_span:
                    span_msg = (
                        f"{beat.beat_id} spans {cur_max - cur_min} pages "
                        f"({all_pages[cur_min]}..{all_pages[cur_max]}), risking visual cramming."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="warning", category="panel_sync", message=span_msg)
                    )
                    flagged_notes.append(span_msg)
                if prev_page_pos is not None and cur_min < prev_page_pos:
                    back_msg = (
                        f"{beat.beat_id} jumped backwards from {all_pages[prev_page_pos]} to {all_pages[cur_min]}."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="panel_sync", message=back_msg)
                    )
                    flagged_notes.append(back_msg)
                if prev_page_pos is not None and (cur_min - prev_page_pos) > max_coverage_gap:
                    gap_msg = (
                        f"Page coverage gap before {beat.beat_id}: jumped from {all_pages[prev_page_pos]} "
                        f"to {all_pages[cur_min]} ({cur_min - prev_page_pos} pages skipped)."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="coverage_gap", message=gap_msg)
                    )
                    flagged_notes.append(gap_msg)
                prev_page_pos = max(prev_page_pos or 0, cur_max)

                for cname, rev_pg, rev_pos, pats in reveal_checks:
                    if rev_pos > cur_max and any(p.search(beat.narration_vi) for p in pats):
                        sp_msg = (
                            f"{beat.beat_id} gọi tên '{cname}' trước khi nhân vật lộ tên trong thoại (tại {rev_pg}) "
                            f"— hãy đổi sang mô tả ngoại hình/vai trò."
                        )
                        issues.append(
                            CriticIssue(beat_id=beat.beat_id, severity="error", category="spoiler_name", message=sp_msg)
                        )
                        flagged_indices.add(idx)
                        flagged_notes.append(sp_msg)

            cur_ng = _content_ngrams(beat.narration_vi)
            if len(cur_ng) >= 8:
                for prev_bid, prev_ng in beat_ngrams:
                    overlap = len(cur_ng & prev_ng)
                    denom = max(1, min(len(cur_ng), len(prev_ng)))
                    if overlap >= 4 and (overlap / denom) >= 0.35:
                        dup_msg = f"{beat.beat_id} possible repeated content overlapping with {prev_bid}."
                        issues.append(
                            CriticIssue(
                                beat_id=beat.beat_id,
                                severity="error",
                                category="duplication",
                                message=dup_msg,
                            )
                        )
                        flagged_indices.add(idx)
                        flagged_notes.append(dup_msg)
                        break
                beat_ngrams.append((beat.beat_id, cur_ng))

            # Adjacent seam echo detector: catches ending of beat i-1 echoed in start of beat i
            if idx > 0:
                prev_tail_w = _tail_head_content_words(beats[idx - 1].narration_vi, is_tail=True, name_tokens=name_tokens)
                cur_head_w = _tail_head_content_words(beat.narration_vi, is_tail=False, name_tokens=name_tokens)
                prev_bg = {f"{prev_tail_w[k]} {prev_tail_w[k + 1]}" for k in range(max(0, len(prev_tail_w) - 1))}
                cur_bg = {f"{cur_head_w[k]} {cur_head_w[k + 1]}" for k in range(max(0, len(cur_head_w) - 1))}
                shared_bg = prev_bg & cur_bg
                shared_single = set(prev_tail_w) & set(cur_head_w)
                if shared_bg or len(shared_single) >= 3:
                    echo_tok = sorted(shared_bg)[0] if shared_bg else ", ".join(sorted(shared_single)[:3])
                    echo_msg = (
                        f"{beat.beat_id} lặp lại ý/từ vừa kết ở {beats[idx - 1].beat_id} ('{echo_tok}') "
                        f"— hãy vào thẳng diễn biến mới, bỏ vế nhắc lại."
                    )
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="warning", category="seam_echo", message=echo_msg)
                    )
                    flagged_indices.add(idx)
                    flagged_notes.append(echo_msg)

            # Collect openers/catchphrases & mid/end-sentence discourse tics (excluding quoted character dialogue)
            for op in extract_beat_openers(beat.narration_vi, name_tokens=name_tokens):
                opener_hits.setdefault(op, []).append(idx)

            narrator_only = _strip_quoted_dialogue(beat.narration_vi)
            for tic_label, tic_pat in _GLOBAL_DISCOURSE_TICS:
                if tic_pat.search(narrator_only):
                    tic_hits.setdefault(tic_label, []).append(idx)
            if ":" in narrator_only:
                colon_beats += 1
            b_sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", beat.narration_vi.strip()) if s.strip()]
            if len(b_sents) >= 3 and (len(beat.narration_vi.split()) / len(b_sents)) <= 9.0:
                staccato_beats += 1

            words = len(beat.narration_vi.split())
            word_counts.append(words)
            if words > 80:
                p_msg = f"{beat.beat_id} exceeds 80 words ({words} từ); tighten phrasing for voiceover."
                issues.append(
                    CriticIssue(beat_id=beat.beat_id, severity="warning", category="pacing", message=p_msg)
                )
                flagged_indices.add(idx)
                flagged_notes.append(p_msg)

            low_text = beat.narration_vi.lower()
            for phrase in _BANNED_CLICHES:
                if phrase in low_text:
                    c_msg = f"{beat.beat_id} chứa cụm sáo rỗng '{phrase}'."
                    issues.append(
                        CriticIssue(beat_id=beat.beat_id, severity="error", category="style", message=c_msg)
                    )
                    flagged_indices.add(idx)
                    flagged_notes.append(c_msg)

        # Opener/Catchphrase Repetition Scanner: keep first 2 occurrences, flag 3rd+ (longest phrase wins per beat)
        repeated_by_beat: dict[int, tuple[str, int]] = {}
        for phrase, hit_indices in sorted(opener_hits.items(), key=lambda kv: (-len(kv[0]), kv[0])):
            uniq_hits = list(dict.fromkeys(hit_indices))
            if len(uniq_hits) >= 3:
                for occ_num, b_idx in enumerate(uniq_hits[2:], start=3):
                    prev_entry = repeated_by_beat.get(b_idx)
                    if prev_entry is None or len(phrase) > len(prev_entry[0]):
                        repeated_by_beat[b_idx] = (phrase, occ_num)
        for b_idx, (phrase, occ_num) in sorted(repeated_by_beat.items()):
            op_msg = (
                f"{beats[b_idx].beat_id} lặp cụm mở/cửa miệng '{phrase}' (lần thứ {occ_num} trong tập) "
                f"— hãy đổi cách dẫn tự nhiên khác (không xóa mất ý phân tích)."
            )
            issues.append(
                CriticIssue(beat_id=beats[b_idx].beat_id, severity="warning", category="opener_repeat", message=op_msg)
            )
            flagged_indices.add(b_idx)
            flagged_notes.append(op_msg)

        # Global Discourse Tic Cap (Critic V3.5): catch overused particles/fillers at ANY position (3rd+ occurrence)
        for tic_label, hit_indices in sorted(tic_hits.items()):
            uniq_hits = list(dict.fromkeys(hit_indices))
            if len(uniq_hits) >= 3:
                for occ_num, b_idx in enumerate(uniq_hits[2:], start=3):
                    t_msg = (
                        f"{beats[b_idx].beat_id} lạm dụng từ đệm/khẩu ngữ '{tic_label}' (lần thứ {occ_num} trong tập) "
                        f"— hãy bỏ từ đệm này và nối câu tự nhiên hơn (giữ nguyên ý và thoại gốc)."
                    )
                    issues.append(
                        CriticIssue(beat_id=beats[b_idx].beat_id, severity="warning", category="tic_repeat", message=t_msg)
                    )
                    flagged_indices.add(b_idx)
                    flagged_notes.append(t_msg)

        # Semantic Visual-Audio Alignment Gate (Quote Pinning + Content Phase-Lag Detection)
        if page_analyses:
            align_groups: list[tuple[list[ScriptBeat], list[str]]] = []
            if part_bounds:
                for pmeta in part_bounds:
                    s_i = int(pmeta.get("start_beat_idx", 0))
                    e_i = int(pmeta.get("end_beat_idx", -1))
                    if 0 <= s_i <= e_i < len(beats):
                        part_pages = list(
                            dict.fromkeys(
                                re.sub(r"_P\d+$", "", pid)
                                for pid in pmeta.get("allowed_panels", [])
                                if pid
                            )
                        )
                        if len(part_pages) >= 3:
                            align_groups.append((beats[s_i : e_i + 1], part_pages))
            elif len(all_pages) >= 3:
                align_groups.append((beats, all_pages))

            for grp_beats, grp_pages in align_groups:
                recs = page_analyses_to_alignment_records(page_analyses, grp_pages)
                for a_iss in verify_beats_alignment(grp_beats, recs):
                    issues.append(
                        CriticIssue(
                            beat_id=a_iss.beat_id,
                            severity="error",
                            category="panel_sync",
                            message=a_iss.message,
                        )
                    )
                    flagged_notes.append(a_iss.message)

        # Macro Distribution Critique (Rule 5): check volume-level pacing variance, staccato/colon overuse & budget
        if len(word_counts) >= 12:
            mean_wc = sum(word_counts) / len(word_counts)
            std_wc = math.sqrt(sum((w - mean_wc) ** 2 for w in word_counts) / len(word_counts))
            if std_wc < 4.5:
                issues.append(
                    CriticIssue(
                        beat_id="global",
                        severity="warning",
                        category="macro_rhythm",
                        message=f"Beat rhythm is overly uniform (std={std_wc:.1f} words); vary short/long beats.",
                    )
                )
            if (colon_beats / len(word_counts)) > 0.30:
                issues.append(
                    CriticIssue(
                        beat_id="global",
                        severity="warning",
                        category="macro_rhythm",
                        message=f"Overused report-style colons ({colon_beats}/{len(word_counts)} beats); weave clauses naturally.",
                    )
                )
            if (staccato_beats / len(word_counts)) > 0.25:
                issues.append(
                    CriticIssue(
                        beat_id="global",
                        severity="warning",
                        category="macro_rhythm",
                        message=f"Excessive choppy staccato beats ({staccato_beats}/{len(word_counts)} beats); connect sentences smoothly.",
                    )
                )

        if target_total_words and len(beats) >= 6:
            actual_words = sum(word_counts)
            if actual_words > int(target_total_words * 1.30):
                excess = actual_words - int(target_total_words * 1.22)
                b_msg = (
                    f"Total script length ({actual_words} từ) exceeds target ({target_total_words} từ) by >30% "
                    f"— hãy rút gọn bớt ~{excess} từ ở các beat dài."
                )
                issues.append(
                    CriticIssue(beat_id="global", severity="error", category="budget", message=b_msg)
                )
                longest_indices = sorted(
                    range(len(beats)),
                    key=lambda i: word_counts[i],
                    reverse=True,
                )[: min(8, len(beats))]
                for li in longest_indices:
                    flagged_indices.add(li)
                flagged_notes.append(b_msg)

        # Series Continuity Gate: glossary/override drift, cross-volume opening echo & seamless hygiene
        if series_ctx is not None and beats:
            gl = getattr(series_ctx, "glossary", None)
            if gl is not None:
                deprecated_map: dict[str, str] = {}
                for raw_k, canon_v in gl.character_name_overrides.items():
                    rk, cv = raw_k.strip(), canon_v.strip()
                    if rk and cv and rk.lower() != cv.lower() and len(rk) >= 2:
                        deprecated_map[rk] = cv
                for raw_k, canon_v in gl.terminology.items():
                    rk, cv = raw_k.strip(), canon_v.strip()
                    # Skip explanatory expansions where canonical form already contains the source term (e.g. "X" -> "X (explanation)")
                    if rk and cv and rk.lower() not in cv.lower() and len(rk) >= 2:
                        deprecated_map[rk] = cv
                for rk, cv in deprecated_map.items():
                    pat = re.compile(rf"(?<!\w){re.escape(rk)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
                    for idx, b in enumerate(beats):
                        if pat.search(b.narration_vi) and cv.lower() not in b.narration_vi.lower():
                            g_msg = (
                                f"{b.beat_id} dùng thuật ngữ/tên chưa đồng nhất '{rk}' — "
                                f"hãy đổi sang bản chuẩn trong glossary: '{cv}'."
                            )
                            issues.append(
                                CriticIssue(
                                    beat_id=b.beat_id,
                                    severity="warning",
                                    category="series_continuity",
                                    message=g_msg,
                                )
                            )
                            flagged_indices.add(idx)
                            flagged_notes.append(g_msg)

            ho = getattr(getattr(series_ctx, "state", None), "last_handoff", None)
            vol_idx = int(getattr(series_ctx, "volume_index", 1) or 1)
            if ho is not None and ho.volume_index < vol_idx and ho.last_beats_tail:
                tail_ngrams: set[str] = set()
                for t_line in ho.last_beats_tail[-2:]:
                    tail_ngrams |= _content_ngrams(t_line)
                if len(tail_ngrams) >= 4:
                    for idx in range(min(2, len(beats))):
                        b_ng = _content_ngrams(beats[idx].narration_vi)
                        if len(b_ng) >= 4:
                            ov = len(b_ng & tail_ngrams) / min(len(b_ng), len(tail_ngrams))
                            if ov >= 0.45:
                                e_msg = (
                                    f"{beats[idx].beat_id} lặp lại gần nguyên văn đoạn kết của tập {ho.volume_index} "
                                    f"(overlap={ov:.0%}) — hãy nối gọn và đi thẳng vào diễn biến mới."
                                )
                                issues.append(
                                    CriticIssue(
                                        beat_id=beats[idx].beat_id,
                                        severity="warning",
                                        category="series_continuity",
                                        message=e_msg,
                                    )
                                )
                                flagged_indices.add(idx)
                                flagged_notes.append(e_msg)

            if getattr(series_ctx, "continuity_mode", "episodic") == "seamless" and vol_idx > 1:
                first_low = beats[0].narration_vi.lower()
                seamless_banned = ("chào mừng", "quay trở lại với tập", "ở tập trước chúng ta")
                for sb in seamless_banned:
                    if sb in first_low:
                        s_msg = (
                            f"{beats[0].beat_id} có lời chào/nhắc tập rời '{sb}' trong chế độ seamless "
                            f"— hãy bỏ câu chào để ghép liền mạch trực tiếp với tập {vol_idx - 1}."
                        )
                        issues.append(
                            CriticIssue(
                                beat_id=beats[0].beat_id,
                                severity="warning",
                                category="series_continuity",
                                message=s_msg,
                            )
                        )
                        flagged_indices.add(0)
                        flagged_notes.append(s_msg)
                        break

        return issues, flagged_indices, flagged_notes

    def _audit_semantic_repetitions(
        self,
        beats: list[ScriptBeat],
        seam_indices: list[int],
    ) -> tuple[set[int], list[str]]:
        """Layer 2 (Read-Only): Scan full script in 1 context ONLY for paraphrase repetition & awkward seams."""
        if not self.polish_seams or not self.llm.has_text_credentials or (len(seam_indices) < 2 and len(beats) < 12):
            return set(), []

        id_to_idx = {b.beat_id: i for i, b in enumerate(beats)}
        seam_ids = [beats[i].beat_id for i in sorted(set(seam_indices)) if 0 <= i < len(beats)]
        script_lines = [
            f"[{b.beat_id} | {b.section}/{b.beat_type}] {b.narration_vi}"
            for b in beats
        ]
        try:
            res = self.llm.complete_xml(
                agent_name="CriticSemanticAuditor",
                system_prompt=(
                    "Bạn là kiểm định viên CHỈ ĐỌC (Read-Only Auditor) cho kịch bản recap manga. "
                    "Nhiệm vụ DUY NHẤT: đọc liền mạch toàn bộ kịch bản và chỉ điểm tối đa 8 beat THỰC SỰ mắc 1 trong 3 lỗi:\n"
                    "1) Lặp ý diễn đạt khác chữ (nhắc lại một chi tiết/nhận định đã nói ở beat trước mà không thêm thông tin mới).\n"
                    "2) Lặp đuôi hồi trước ngay đầu hồi sau tại các điểm nối `seam`.\n"
                    "3) Lặp câu cửa miệng / mẫu câu hỏi quá lộ (ví dụ mở câu giống nhau 3-4 lần).\n"
                    "TUYỆT ĐỐI KHÔNG gắn cờ câu hỏi giao lưu ('Anh em...', 'Thử nghĩ xem...'), câu bình luận hay, hay beat ngắn dồn dập. "
                    "Nếu kịch bản đã sạch lặp, trả về `<audit></audit>` rỗng."
                ),
                user_prompt=(
                    f"Các điểm nối hồi (seam): {seam_ids}\n\n"
                    f"Toàn bộ kịch bản ({len(beats)} beats):\n" + "\n".join(script_lines)
                ),
                xml_template=_SEMANTIC_AUDIT_XML_TEMPLATE,
                parse_fn=_parse_semantic_audit_xml,
            )
            if not isinstance(res, _SemanticAuditResponse):
                return set(), []
            found_indices: set[int] = set()
            found_notes: list[str] = []
            for fl in res.flags[:8]:
                idx = id_to_idx.get(fl.beat_id)
                if idx is not None and idx not in found_indices:
                    found_indices.add(idx)
                    found_notes.append(f"{fl.beat_id} [audit lặp ý/seam]: {fl.reason}")
            return found_indices, found_notes
        except Exception:
            return set(), []

    @staticmethod
    def _is_safe_beat_rewrite(
        beat_idx: int,
        old_text: str,
        new_text: str,
        beats: list[ScriptBeat],
        page_order: dict[str, int],
        reveal_checks: list[tuple[str, str, int, list[re.Pattern[str]]]],
        beat_issue_cats: set[str],
        repeated_phrases: set[str],
    ) -> bool:
        """Per-beat rollback guard: accept rewrite only if clean, spoiler-free, and doesn't worsen duplication."""
        txt = new_text.strip()
        if not txt or txt == old_text.strip() or txt[0].islower():
            return False
        low = txt.lower()
        if any(phrase in low for phrase in _BANNED_CLICHES):
            return False

        # Reject if rewrite still keeps the exact overused opener/tic it was supposed to replace
        if (beat_issue_cats & {"opener_repeat", "tic_repeat"}) and repeated_phrases:
            narrator_low = _strip_quoted_dialogue(txt).lower()
            if any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", narrator_low, flags=re.UNICODE) for p in repeated_phrases):
                return False

        # Reject if rewrite introduces a new character name spoiler before first_named_page_id
        positions = [p for pid in beats[beat_idx].panel_ids if (p := _page_pos_from_panel(pid, page_order)) is not None]
        if positions and beats[beat_idx].beat_type != "hook_montage":
            cur_max = max(positions)
            for _, _, rev_pos, pats in reveal_checks:
                if rev_pos > cur_max and any(p.search(txt) for p in pats):
                    return False

        # Preserve verbatim quoted dialogue unless beat is being trimmed for budget/duplication/spoiler
        if (
            not (beat_issue_cats & {"budget", "duplication", "spoiler_name"})
            and _has_dialogue_quote(old_text)
            and not _has_dialogue_quote(txt)
            and len(txt.split()) < len(old_text.split()) * 0.65
        ):
            return False

        # Reject if rewrite creates brand-new heavy N-gram duplication with any other beat
        new_ng = _content_ngrams(txt)
        if len(new_ng) >= 8:
            for j, other in enumerate(beats):
                if j == beat_idx:
                    continue
                other_ng = _content_ngrams(other.narration_vi)
                if len(other_ng) < 8:
                    continue
                ov = len(new_ng & other_ng)
                denom = max(1, min(len(new_ng), len(other_ng)))
                if ov >= 4 and (ov / denom) >= 0.35:
                    return False

        return True

    def _polish_multi_part_seams(
        self,
        beats: list[ScriptBeat],
        bible: StoryBible,
        seam_indices: list[int],
        flagged_indices: set[int],
        flagged_notes: list[str],
        panel_text_lookup: dict[str, str] | None = None,
        run_global_showrunner: bool = False,
        target_total_words: int | None = None,
        persona: str = "kich_tinh",
        page_order: dict[str, int] | None = None,
        beat_issues_map: dict[int, list[CriticIssue]] | None = None,
    ) -> list[ScriptBeat]:
        """Layer 3 (Surgical Repairer): Edit ONLY allowlisted flagged beats with read-only neighbor context & rollback."""
        if not self.polish_seams or not self.llm.has_text_credentials or len(beats) < 6:
            return beats

        editable_indices = sorted(i for i in flagged_indices if 0 <= i < len(beats))
        if not editable_indices:
            return beats

        editable_set = set(editable_indices)
        editable_beat_ids = {beats[i].beat_id for i in editable_indices}

        # Include immediate neighbor beats strictly as read-only context so tone/flow stay seamless
        context_indices: set[int] = set(editable_set)
        for fi in editable_indices:
            if fi - 1 >= 0:
                context_indices.add(fi - 1)
            if fi + 1 < len(beats):
                context_indices.add(fi + 1)
        ordered_indices = sorted(context_indices)

        issues_by_idx = beat_issues_map or {}
        payload_lines = ["<beats>"]
        for i in ordered_indices:
            b = beats[i]
            is_ed = i in editable_set
            mode_attr = 'editable="true"' if is_ed else 'readonly="true"'
            extra_blocks = ""
            if is_ed:
                if panel_text_lookup:
                    facts = [panel_text_lookup[pid] for pid in b.panel_ids if pid in panel_text_lookup]
                    if facts:
                        facts_txt = " | ".join(facts).replace('"', "'")
                        extra_blocks += f"\n    <grounded_facts>{facts_txt}</grounded_facts>"
                b_msgs = [iss.message for iss in issues_by_idx.get(i, [])]
                if b_msgs:
                    extra_blocks += f"\n    <fix_reason>{' ; '.join(b_msgs)}</fix_reason>"
            payload_lines.append(
                f'  <beat id="{b.beat_id}" {mode_attr} section="{b.section}" beat_type="{b.beat_type}" mood="{b.mood}">\n'
                f"    <narration>{b.narration_vi}</narration>{extra_blocks}\n"
                "  </beat>"
            )
        payload_lines.append("</beats>")

        cast_names = [
            f"{c.canonical_name} (lộ từ {c.first_named_page_id})" if c.first_named_page_id else c.canonical_name
            for c in bible.characters
        ]
        notes_txt = "\n".join(f"- {n}" for n in flagged_notes)
        budget_hint = (
            f"Tổng ngân sách mục tiêu toàn tập: ~{target_total_words} từ (chỉ rút gọn ở các beat editable bị gắn cờ budget)."
            if target_total_words and any("exceeds target" in n for n in flagged_notes)
            else ""
        )
        personas_cfg = load_personas()
        persona_name = personas_cfg.get(persona, {}).get("name", persona)
        persona_voice = (personas_cfg.get(persona, {}).get("system_prompt", "") or "").splitlines()[0].strip()

        try:
            res = self.llm.complete_xml(
                agent_name="CriticSeamPolisher",
                system_prompt=(
                    f"Bạn là Bác Sĩ Phẫu Thuật Kịch Bản Voiceover cho persona '{persona}' ({persona_name}).\n"
                    f"Giọng chuẩn của persona: {persona_voice}\n"
                    "QUY TẮC TUYỆT ĐỐI:\n"
                    "1) CHỈ trả về `<rewrite beat_id='...'>` cho những beat có thuộc tính `editable=\"true\"`. TUYỆT ĐỐI KHÔNG sửa beat `readonly=\"true\"`.\n"
                    "2) CHỈ sửa đúng lỗi nêu trong `<fix_reason>` / danh sách lỗi (đổi cụm mở đầu bị lặp, bỏ vế lặp đuôi beat trước, bỏ tên gọi sớm trước `lộ từ ...`, khử cliché AI, hoặc cắt ngắn khi vượt budget).\n"
                    "3) GIỮ NGUYÊN 100% sự thật từ `<grounded_facts>`, câu thoại trích dẫn trong ngoặc kép (\"...\"), đúng giọng kể tự nhiên của persona (KHÔNG ép khẩu ngữ thừa) và nhịp ngắn-dài của bản gốc."
                ),
                user_prompt=(
                    f"Nhân vật chuẩn: {cast_names}\n"
                    f"Danh sách beat ĐƯỢC PHÉP SỬA (allowlist): {sorted(editable_beat_ids)}\n"
                    f"{budget_hint}\n"
                    f"Các lỗi cần phẫu thuật:\n{notes_txt}\n\n"
                    f"Ngữ cảnh kịch bản ({len(editable_indices)} beat editable + beat kề readonly):\n"
                    + "\n".join(payload_lines)
                ),
                xml_template=_SEAM_POLISH_XML_TEMPLATE,
                parse_fn=_parse_seam_polish_xml,
            )

            p_order = page_order or {}
            reveal_checks = self._build_reveal_checks(bible, p_order)
            id_to_idx = {b.beat_id: idx for idx, b in enumerate(beats)}

            for r in res.rewrites:
                # Strict Python allowlist: ignore any rewrite targeting a clean beat
                if r.beat_id not in editable_beat_ids:
                    continue
                b_idx = id_to_idx.get(r.beat_id)
                if b_idx is None:
                    continue
                b_issues = issues_by_idx.get(b_idx, [])
                cats = {iss.category for iss in b_issues}
                rep_phrases: set[str] = set()
                for iss in b_issues:
                    if iss.category in ("opener_repeat", "tic_repeat"):
                        m = re.search(r"'([^']+)'", iss.message)
                        if m:
                            rep_phrases.add(m.group(1).lower())
                if self._is_safe_beat_rewrite(
                    beat_idx=b_idx,
                    old_text=beats[b_idx].narration_vi,
                    new_text=r.narration_vi,
                    beats=beats,
                    page_order=p_order,
                    reveal_checks=reveal_checks,
                    beat_issue_cats=cats,
                    repeated_phrases=rep_phrases,
                ):
                    beats[b_idx].narration_vi = r.narration_vi.strip()
        except Exception:
            pass
        return beats

    def _repair_panel_alignment(
        self,
        beats: list[ScriptBeat],
        issues_eval: list[CriticIssue],
        page_analyses: list[PageAnalysis],
        all_pages: list[str],
        page_order: dict[str, int],
        valid_ids: list[str],
        part_bounds: list[dict],
    ) -> list[ScriptBeat]:
        """Layer 3 (Visual): Surgical panel_ids repair (LLM allowlist + rollback guard + deterministic safety-net)."""
        panel_err_ids = {
            iss.beat_id
            for iss in issues_eval
            if iss.severity == "error" and iss.category in ("panel_sync", "coverage_gap") and iss.beat_id != "global"
        }
        if not panel_err_ids or not page_analyses:
            return beats

        groups: list[tuple[int, int, list[str], list[str]]] = []
        if part_bounds:
            for pmeta in part_bounds:
                s_i = int(pmeta.get("start_beat_idx", 0))
                e_i = int(pmeta.get("end_beat_idx", -1))
                if not (0 <= s_i <= e_i < len(beats)):
                    continue
                allowed_p = [pid for pid in pmeta.get("allowed_panels", []) if pid]
                part_pgs = list(dict.fromkeys(re.sub(r"_P\d+$", "", pid) for pid in allowed_p))
                if len(part_pgs) >= 3 and any(beats[i].beat_id in panel_err_ids for i in range(s_i, e_i + 1)):
                    groups.append((s_i, e_i, part_pgs, allowed_p))
        elif len(all_pages) >= 3:
            groups.append((0, len(beats) - 1, all_pages, valid_ids))

        def _count_group_errors(grp: list[ScriptBeat], recs: list) -> tuple[int, int]:
            align_errs = len(verify_beats_alignment(grp, recs))
            back_errs = 0
            prev_p: int | None = None
            for b in grp:
                if b.beat_type == "hook_montage":
                    continue
                pos = [p for pid in b.panel_ids if (p := _page_pos_from_panel(pid, page_order)) is not None]
                if pos:
                    if prev_p is not None and min(pos) < prev_p:
                        back_errs += 1
                    prev_p = max(prev_p or 0, max(pos))
            return align_errs, back_errs

        for s_i, e_i, grp_pages, allowed_p in groups:
            grp_beats = beats[s_i : e_i + 1]
            recs = page_analyses_to_alignment_records(page_analyses, grp_pages)
            if len(recs) < 3:
                continue
            allowed_set = set(allowed_p)
            panel_rank = {pid: i for i, pid in enumerate(allowed_p)}
            before_align, before_back = _count_group_errors(grp_beats, recs)
            if before_align == 0 and before_back == 0:
                continue

            # Tier 1: Allowlisted LLM Surgical Panel Re-Aligner with strict rollback guard
            if self.polish_seams and self.llm.has_text_credentials:
                local_err_idxs = {
                    i for i, b in enumerate(grp_beats) if b.beat_id in panel_err_ids
                }
                editable_local: set[int] = set()
                for li in local_err_idxs:
                    for nb in (li - 1, li, li + 1):
                        if 0 <= nb < len(grp_beats):
                            editable_local.add(nb)
                editable_bids = {grp_beats[i].beat_id for i in editable_local}
                anchor_md = build_alignment_anchors(" ".join(b.narration_vi for b in grp_beats), recs)

                beat_lines = []
                for i, b in enumerate(grp_beats):
                    mode = 'editable="true"' if i in editable_local else 'readonly="true"'
                    msgs = [iss.message for iss in issues_eval if iss.beat_id == b.beat_id and iss.category in ("panel_sync", "coverage_gap")]
                    reason_xml = f"\n    <fix_reason>{' ; '.join(msgs)}</fix_reason>" if msgs else ""
                    beat_lines.append(
                        f'  <beat id="{b.beat_id}" {mode}>\n'
                        f"    <panels>{', '.join(b.panel_ids)}</panels>\n"
                        f"    <narration>{b.narration_vi}</narration>{reason_xml}\n"
                        "  </beat>"
                    )
                page_lines = []
                for rec in recs:
                    p_str = " ; ".join(f"[{pid}] {txt[:100]}" for pid, txt in rec.panels)
                    page_lines.append(f"- {rec.page_id}: {p_str}")

                try:
                    res = self.llm.complete_xml(
                        agent_name="CriticPanelRepairer",
                        system_prompt=(
                            "Bạn là Kỹ Sư Phẫu Thuật Đồng Bộ Hình–Tiếng. "
                            "CHỈ sửa danh sách `<panels>` của các beat có `editable=\"true\"` để khớp chính xác hành động và lời thoại gốc, "
                            "tiến đơn điệu từ đầu tới cuối hồi. TUYỆT ĐỐI KHÔNG sửa beat `readonly=\"true\"` và KHÔNG viết lại lời đọc."
                        ),
                        user_prompt=(
                            f"Allowlist beat được phép sửa panel: {sorted(editable_bids)}\n\n"
                            f"### BẢN ĐỒ NEO TRANG:\n{anchor_md}\n\n"
                            f"### DANH SÁCH TRANG & PANEL HỢP LỆ:\n" + "\n".join(page_lines) + "\n\n"
                            f"### CÁC BEAT TRONG HỒI:\n<beats>\n" + "\n".join(beat_lines) + "\n</beats>\n\n"
                            "Trả về XML `<panel_repairs>` chứa 1–4 `panel_id` hợp lệ cho từng beat cần chỉnh."
                        ),
                        xml_template=_PANEL_REPAIR_XML_TEMPLATE,
                        parse_fn=_parse_panel_repair_xml,
                        temperature_override=0.15,
                    )
                    if res.repairs:
                        cand_grp = [b.model_copy(deep=True) for b in grp_beats]
                        bid_map = {b.beat_id: idx for idx, b in enumerate(cand_grp)}
                        applied = False
                        for rep_item in res.repairs:
                            if rep_item.beat_id not in editable_bids:
                                continue
                            li = bid_map.get(rep_item.beat_id)
                            if li is None:
                                continue
                            valid_new = [pid for pid in rep_item.panel_ids if pid in allowed_set]
                            valid_new = list(dict.fromkeys(sorted(valid_new, key=lambda p: panel_rank.get(p, 0))))[:4]
                            if valid_new and valid_new != cand_grp[li].panel_ids:
                                cand_grp[li].panel_ids = valid_new
                                applied = True
                        if applied:
                            after_align, after_back = _count_group_errors(cand_grp, recs)
                            if after_back <= before_back and (after_align + after_back) < (before_align + before_back):
                                for offset, cb in enumerate(cand_grp):
                                    beats[s_i + offset].panel_ids = cb.panel_ids
                                grp_beats = beats[s_i : e_i + 1]
                except Exception:
                    pass

            # Tier 2: Deterministic monotonic safety-net for any residual quote/desync issue
            repair_part_panel_alignment_deterministic(grp_beats, recs, allowed_p)

        return beats

    def review(
        self,
        script: RecapScript,
        manifest: JobManifest,
        bible: StoryBible,
        job_dir: Path,
        force: bool = False,
        script_config: ScriptConfig | None = None,
        page_analyses: list[PageAnalysis] | None = None,
        series_ctx=None,
    ) -> tuple[RecapScript, CriticReport]:
        job_dir.mkdir(parents=True, exist_ok=True)
        final_path = job_dir / "03b_script_final.json"
        report_path = job_dir / "03b_critic_report.json"
        if final_path.exists() and report_path.exists() and not force:
            return (
                RecapScript.model_validate_json(final_path.read_text(encoding="utf-8")),
                CriticReport.model_validate_json(report_path.read_text(encoding="utf-8")),
            )

        all_pages = [pg.page_id for ch in manifest.chapters for pg in ch.pages]
        page_order = {pg_id: i for i, pg_id in enumerate(all_pages)}
        valid_ids = [
            p.panel_id
            for ch in manifest.chapters
            for pg in ch.pages
            for p in pg.panels
        ]
        valid_set = set(valid_ids)

        panel_text_lookup: dict[str, str] = {}
        if page_analyses:
            for pa in page_analyses:
                for pnl in pa.panels:
                    dlg = f' "{pnl.dialogue_summary}"' if pnl.dialogue_summary else ""
                    panel_text_lookup[pnl.panel_id] = f"{pnl.action_summary}{dlg}".strip()

        seam_indices: list[int] = []
        beat_to_allowed: dict[int, set[str]] = {}
        part_bounds: list[dict] = []
        boundaries_path = job_dir / "repo" / "script" / "part_boundaries.json"
        if boundaries_path.exists():
            try:
                bounds = json.loads(boundaries_path.read_text(encoding="utf-8"))
                part_bounds = bounds if isinstance(bounds, list) else []
                for b_idx, part_meta in enumerate(part_bounds):
                    end_i = int(part_meta.get("end_beat_idx", -1))
                    if b_idx < len(part_bounds) - 1 and end_i >= 0:
                        seam_indices.extend([end_i, end_i + 1])
                    allowed = set(part_meta.get("allowed_panels", []))
                    if allowed:
                        for idx_in in range(
                            int(part_meta.get("start_beat_idx", 0)),
                            int(part_meta.get("end_beat_idx", -1)) + 1,
                        ):
                            beat_to_allowed[idx_in] = allowed
            except Exception:
                seam_indices = []
                part_bounds = []
        seam_indices = sorted(set(seam_indices))

        # Step 1: Deterministic Panel Grounding repair (guaranteed valid panel_ids)
        grounding_issues: list[CriticIssue] = []
        cleaned_beats: list[ScriptBeat] = []
        total_beats = len(script.beats)
        for idx, beat in enumerate(script.beats):
            b_copy = beat.model_copy(deep=True)
            kept = [pid for pid in b_copy.panel_ids if pid in valid_set]
            if not kept and valid_ids:
                fallback_id = self._scoped_fallback_panel(idx, total_beats, valid_ids, cleaned_beats)
                kept = [fallback_id]
                grounding_issues.append(
                    CriticIssue(
                        beat_id=b_copy.beat_id,
                        severity="warning",
                        category="grounding",
                        message=f"Replaced invalid panel_ids with scoped {fallback_id}.",
                    )
                )
            b_copy.panel_ids = kept
            cleaned_beats.append(b_copy)

        target_total_words = (
            max(400, int(round(script_config.target_duration_min * script_config.words_per_minute)))
            if script_config is not None and len(cleaned_beats) >= 6
            else None
        )

        # Layer 1: Deterministic Gate + Semantic Visual-Audio Alignment + Repetition Scanner + Series Continuity
        issues_eval, flagged_indices, flagged_notes = self._evaluate_issues(
            cleaned_beats,
            all_pages=all_pages,
            page_order=page_order,
            beat_to_allowed=beat_to_allowed,
            target_total_words=target_total_words,
            bible=bible,
            page_analyses=page_analyses,
            part_bounds=part_bounds,
            series_ctx=series_ctx,
        )

        # Layer 3a (Visual): Surgical Panel Repair when alignment/coverage errors survive Stage 3a
        if page_analyses and any(
            i.severity == "error" and i.category in ("panel_sync", "coverage_gap") for i in issues_eval
        ):
            cleaned_beats = self._repair_panel_alignment(
                cleaned_beats,
                issues_eval=issues_eval,
                page_analyses=page_analyses,
                all_pages=all_pages,
                page_order=page_order,
                valid_ids=valid_ids,
                part_bounds=part_bounds,
            )
            issues_eval, flagged_indices, flagged_notes = self._evaluate_issues(
                cleaned_beats,
                all_pages=all_pages,
                page_order=page_order,
                beat_to_allowed=beat_to_allowed,
                target_total_words=target_total_words,
                bible=bible,
                page_analyses=page_analyses,
                part_bounds=part_bounds,
                series_ctx=series_ctx,
            )

        # Layer 2 (Read-Only Semantic Auditor) + Layer 3 (Allowlisted Surgical Repairer, up to 2 rounds)
        if self.polish_seams and self.llm.has_text_credentials and len(cleaned_beats) >= 6:
            id_to_idx = {b.beat_id: i for i, b in enumerate(cleaned_beats)}
            audit_indices, audit_notes = self._audit_semantic_repetitions(cleaned_beats, seam_indices)

            beat_issues_map: dict[int, list[CriticIssue]] = {}
            for iss in issues_eval:
                if iss.category in _TEXT_FIXABLE_CATEGORIES and iss.beat_id in id_to_idx:
                    beat_issues_map.setdefault(id_to_idx[iss.beat_id], []).append(iss)
            for anote in audit_notes:
                abid = anote.split(" ", 1)[0]
                if abid in id_to_idx:
                    beat_issues_map.setdefault(id_to_idx[abid], []).append(
                        CriticIssue(beat_id=abid, severity="warning", category="semantic_repeat", message=anote)
                    )
            if any(iss.category == "budget" for iss in issues_eval):
                b_iss = next(iss for iss in issues_eval if iss.category == "budget")
                for fi in flagged_indices:
                    beat_issues_map.setdefault(fi, []).append(b_iss)

            editable_round1 = flagged_indices | audit_indices
            if editable_round1:
                cleaned_beats = self._polish_multi_part_seams(
                    cleaned_beats,
                    bible,
                    seam_indices=seam_indices,
                    flagged_indices=editable_round1,
                    flagged_notes=flagged_notes + audit_notes,
                    panel_text_lookup=panel_text_lookup,
                    run_global_showrunner=False,
                    target_total_words=target_total_words,
                    persona=script.persona,
                    page_order=page_order,
                    beat_issues_map=beat_issues_map,
                )
                issues_eval, flagged_indices, flagged_notes = self._evaluate_issues(
                    cleaned_beats,
                    all_pages=all_pages,
                    page_order=page_order,
                    beat_to_allowed=beat_to_allowed,
                    target_total_words=target_total_words,
                    bible=bible,
                    page_analyses=page_analyses,
                    part_bounds=part_bounds,
                    series_ctx=series_ctx,
                )

                # Round 2: ONLY if text-fixable severity="error" remains (never waste LLM calls on structural errors)
                remaining_text_errors = [
                    i for i in issues_eval if i.severity == "error" and i.category in _TEXT_FIXABLE_CATEGORIES
                ]
                if remaining_text_errors:
                    retry_map: dict[int, list[CriticIssue]] = {}
                    retry_indices: set[int] = set()
                    for iss in remaining_text_errors:
                        if iss.beat_id in id_to_idx:
                            bi = id_to_idx[iss.beat_id]
                            retry_indices.add(bi)
                            retry_map.setdefault(bi, []).append(iss)
                        elif iss.category == "budget":
                            for fi in flagged_indices:
                                retry_indices.add(fi)
                                retry_map.setdefault(fi, []).append(iss)
                    if retry_indices:
                        cleaned_beats = self._polish_multi_part_seams(
                            cleaned_beats,
                            bible,
                            seam_indices=[],
                            flagged_indices=retry_indices,
                            flagged_notes=[i.message for i in remaining_text_errors],
                            panel_text_lookup=panel_text_lookup,
                            run_global_showrunner=False,
                            target_total_words=target_total_words,
                            persona=script.persona,
                            page_order=page_order,
                            beat_issues_map=retry_map,
                        )
                        issues_eval, _, _ = self._evaluate_issues(
                            cleaned_beats,
                            all_pages=all_pages,
                            page_order=page_order,
                            beat_to_allowed=beat_to_allowed,
                            target_total_words=target_total_words,
                            bible=bible,
                            page_analyses=page_analyses,
                            part_bounds=part_bounds,
                            series_ctx=series_ctx,
                        )

        # Step 3: Populate TTS pronunciation-normalized strings without mutating display subtitles
        for b in cleaned_beats:
            b.tts_text = apply_pronunciation_glossary(b.narration_vi, bible.pronunciation_map)

        all_issues = grounding_issues + issues_eval
        final_script = RecapScript(
            title_vi=script.title_vi,
            persona=script.persona,
            hook_strategy=script.hook_strategy,
            hook_summary=script.hook_summary,
            beats=cleaned_beats,
        )
        err_count = sum(1 for i in all_issues if i.severity == "error")
        warn_count = len(all_issues) - err_count
        report = CriticReport(
            passed=(err_count == 0),
            score=max(2.0, round(10.0 - 1.2 * err_count - 0.35 * warn_count, 2)),
            issues=all_issues,
            notes="Validated hard invariants, opener/seam repetition, Read-Only Audit, allowlisted Surgical Repair, and TTS pronunciation.",
        )
        final_path.write_text(final_script.model_dump_json(indent=2), encoding="utf-8")
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return final_script, report
