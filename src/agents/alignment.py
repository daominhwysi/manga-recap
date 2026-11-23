"""Semantic & Quote-Pinned Panel Alignment (pre-alignment anchor hints + post-alignment verification).

Series-agnostic, pure-Python statistical alignment using per-part IDF unigrams/bigrams,
exact quoted-dialogue pinning, and monotonic dynamic programming.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re

from src.schemas.analysis import PageAnalysis
from src.schemas.script import ScriptBeat


@dataclass
class PageAlignmentRecord:
    """Normalized page representation for beat-to-page alignment checks."""

    page_id: str
    full_text: str
    panels: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class AlignmentIssue:
    """Detected mismatch between a beat's narration and its assigned panel_ids."""

    beat_idx: int
    beat_id: str
    kind: str  # "quote_mismatch" | "visual_desync"
    expected_page_id: str
    expected_panel_id: str | None
    assigned_pages: list[str]
    message: str


def _panel_to_page_id(panel_id: str) -> str:
    return re.sub(r"_P\d+$", "", panel_id)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE) if len(w) >= 2]


def _extract_features(text: str) -> Counter[str]:
    toks = _tokenize(text)
    feats: Counter[str] = Counter(toks)
    for i in range(len(toks) - 1):
        feats[f"{toks[i]}_{toks[i + 1]}"] += 1
    return feats


def _extract_quoted_spans(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r'["“]([^"”]{8,})["”]', text)]


def _split_prose_sentences(prose: str) -> list[str]:
    """Split prose into sentences without breaking inside quoted dialogue."""
    text = re.sub(r"\s+", " ", prose).strip()
    if not text:
        return []
    sentences: list[str] = []
    buf: list[str] = []
    in_quote = False
    n = len(text)
    for i, ch in enumerate(text):
        buf.append(ch)
        if ch in ('"', "“", "”"):
            if ch == "“":
                in_quote = True
            elif ch == "”":
                in_quote = False
            else:
                in_quote = not in_quote
        if not in_quote and ch in ".!?…":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in ("", " ", "\n"):
                s = "".join(buf).strip()
                if s:
                    sentences.append(s)
                buf = []
    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def parse_scene_pages_for_alignment(scene_md: str) -> list[PageAlignmentRecord]:
    """Parse compressed scene markdown (repo/notes/scene_*.md) into PageAlignmentRecords."""
    pages: list[PageAlignmentRecord] = []
    cur: PageAlignmentRecord | None = None
    for raw_line in scene_md.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        hm = re.match(r"^###\s+([a-zA-Z0-9_]+):\s*(.*)$", line)
        if hm:
            if cur is not None:
                pages.append(cur)
            cur = PageAlignmentRecord(page_id=hm.group(1), full_text=hm.group(2), panels=[])
            continue
        if cur is not None and line.startswith("- ["):
            pm = re.match(r"^-\s+\[([^\]]+)\]\s*(?:\([^)]*\)\s*)?(.*)$", line)
            if pm:
                pid, rest = pm.group(1).strip(), pm.group(2).strip()
                cur.full_text = f"{cur.full_text} {rest}".strip()
                cur.panels.append((pid, rest))
    if cur is not None:
        pages.append(cur)
    return pages


def page_analyses_to_alignment_records(
    page_analyses: list[PageAnalysis],
    allowed_page_ids: list[str] | None = None,
) -> list[PageAlignmentRecord]:
    """Convert Stage 2a PageAnalysis list into ordered PageAlignmentRecords."""
    by_id = {pa.page_id: pa for pa in page_analyses}
    ordered_ids = allowed_page_ids if allowed_page_ids is not None else [pa.page_id for pa in page_analyses]
    records: list[PageAlignmentRecord] = []
    for pg_id in ordered_ids:
        pa = by_id.get(pg_id)
        if not pa:
            continue
        parts = [pa.page_summary]
        if pa.verbatim_quote_vi:
            parts.append(pa.verbatim_quote_vi)
        if pa.visual_punch:
            parts.append(pa.visual_punch)
        panels: list[tuple[str, str]] = []
        for pnl in pa.panels:
            ptxt = f"{pnl.action_summary} {pnl.dialogue_summary}".strip()
            panels.append((pnl.panel_id, ptxt))
            if ptxt:
                parts.append(ptxt)
        records.append(
            PageAlignmentRecord(
                page_id=pa.page_id,
                full_text=" ".join(p for p in parts if p).strip(),
                panels=panels,
            )
        )
    return records


class _PartAlignmentIndex:
    """Precomputed per-part IDF feature table & quote matcher."""

    def __init__(self, pages: list[PageAlignmentRecord]) -> None:
        self.pages = pages
        self.page_idx = {pg.page_id: i for i, pg in enumerate(pages)}
        self.page_feats = [_extract_features(pg.full_text) for pg in pages]
        df: Counter[str] = Counter()
        for pf in self.page_feats:
            for k in pf:
                df[k] += 1
        n_pg = max(1, len(pages))
        self.idf = {k: math.log((n_pg + 1.0) / (v + 1.0)) + 0.5 for k, v in df.items()}

    def score_text_on_page(self, text: str, pg_i: int) -> tuple[float, int, int]:
        """Return (idf_similarity, matched_unigrams, matched_bigrams)."""
        tf = _extract_features(text)
        if not tf:
            return 0.0, 0, 0
        pf = self.page_feats[pg_i]
        num = 0.0
        den = 0.0
        u_hits = 0
        b_hits = 0
        for k in tf:
            is_bi = "_" in k
            w = self.idf.get(k, 0.2) * (1.5 if is_bi else 1.0)
            den += w
            if k in pf:
                num += w
                if is_bi:
                    b_hits += 1
                else:
                    u_hits += 1
        return num / max(1e-6, den), u_hits, b_hits

    def find_quote_pin(self, quote: str) -> tuple[float, int, str] | None:
        """Locate the exact (score, page_idx, panel_id) for a quoted dialogue span."""
        q_toks = _tokenize(quote)
        if len(q_toks) < 4:
            return None
        q_set = set(q_toks)
        q_bigs = {f"{q_toks[i]}_{q_toks[i + 1]}" for i in range(len(q_toks) - 1)}

        candidates: list[tuple[float, int, str]] = []
        for pg_i, pg in enumerate(self.pages):
            best_on_pg = (0.0, pg_i, pg.panels[0][0] if pg.panels else "")
            for pid, ptxt in pg.panels:
                p_toks = _tokenize(ptxt)
                if not p_toks:
                    continue
                p_set = set(p_toks)
                p_bigs = {f"{p_toks[i]}_{p_toks[i + 1]}" for i in range(len(p_toks) - 1)}
                u_ov = len(q_set & p_set) / len(q_set)
                b_ov = (len(q_bigs & p_bigs) / len(q_bigs)) if q_bigs else u_ov
                sc = 0.45 * u_ov + 0.55 * b_ov
                if sc > best_on_pg[0]:
                    best_on_pg = (sc, pg_i, pid)
            # Also check page-level summary / verbatim quote if panels didn't capture full sentence
            pg_toks = _tokenize(pg.full_text)
            if pg_toks:
                pg_set = set(pg_toks)
                pg_bigs = {f"{pg_toks[i]}_{pg_toks[i + 1]}" for i in range(len(pg_toks) - 1)}
                u_ov = len(q_set & pg_set) / len(q_set)
                b_ov = (len(q_bigs & pg_bigs) / len(q_bigs)) if q_bigs else u_ov
                pg_sc = 0.45 * u_ov + 0.55 * b_ov
                if pg_sc > best_on_pg[0] + 0.08:
                    best_on_pg = (pg_sc, pg_i, best_on_pg[2])
            if best_on_pg[0] >= 0.62 and best_on_pg[2]:
                candidates.append(best_on_pg)

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[0]
        # Avoid ambiguous pins when two distant pages contain the exact same recurring catchphrase
        if len(candidates) >= 2 and abs(candidates[1][1] - top[1]) >= 2 and (top[0] - candidates[1][0]) < 0.08:
            return None
        return top


def build_alignment_anchors(source_prose: str, pages: list[PageAlignmentRecord]) -> str:
    """Build a compact monotonic sentence-to-page anchor map to guide Pass 2 PanelAlignerAgent."""
    if len(pages) < 2:
        return ""
    sentences = _split_prose_sentences(source_prose)
    if not sentences:
        return ""

    index = _PartAlignmentIndex(pages)
    n_sent = len(sentences)
    n_pg = len(pages)

    # Compute emission matrix + quote pins per sentence
    emissions: list[list[float]] = []
    sent_pins: list[list[tuple[int, str, str]]] = []
    for s in sentences:
        row = [index.score_text_on_page(s, j)[0] for j in range(n_pg)]
        pins: list[tuple[int, str, str]] = []
        for q in _extract_quoted_spans(s):
            pin = index.find_quote_pin(q)
            if pin:
                _, q_pgi, q_pid = pin
                row[q_pgi] += 2.5
                pins.append((q_pgi, q_pid, q))
        emissions.append(row)
        sent_pins.append(pins)

    # Monotonic Viterbi DP: choose non-decreasing page index p_0 <= p_1 <= ... <= p_{n_sent-1}
    dp = [[-1e9] * n_pg for _ in range(n_sent)]
    back = [[0] * n_pg for _ in range(n_sent)]
    for j in range(n_pg):
        # Gentle prior encouraging start near early pages when sentence 0 has weak lexical signal
        prior = -0.02 * j
        dp[0][j] = emissions[0][j] + prior
    for i in range(1, n_sent):
        expected_step = n_pg / max(1.0, float(n_sent))
        for j in range(n_pg):
            best_val = -1e9
            best_k = 0
            for k in range(j + 1):
                step = j - k
                # Small preference for forward progression; never block genuine scene jumps
                trans_pen = -0.008 * max(0.0, step - expected_step * 2.2) - (0.015 if step == 0 else 0.0)
                val = dp[i - 1][k] + emissions[i][j] + trans_pen
                if val > best_val:
                    best_val = val
                    best_k = k
            dp[i][j] = best_val
            back[i][j] = best_k

    # Backtrack best monotonic path
    best_last = max(range(n_pg), key=lambda j: dp[n_sent - 1][j])
    path = [0] * n_sent
    path[n_sent - 1] = best_last
    for i in range(n_sent - 1, 0, -1):
        path[i - 1] = back[i][path[i]]

    lines: list[str] = []
    for i, s in enumerate(sentences):
        anchor_j = path[i]
        # Include top secondary matching page within [anchor_j-1 .. min(n_pg-1, next_anchor)] for multi-page summary sentences
        hi_j = path[i + 1] if i + 1 < n_sent else n_pg - 1
        lo_j = path[i - 1] if i > 0 else 0
        strong_pages = [
            j for j in range(lo_j, hi_j + 1)
            if emissions[i][j] >= max(0.24, emissions[i][anchor_j] * 0.72)
        ]
        if anchor_j not in strong_pages:
            strong_pages.append(anchor_j)
        strong_pages.sort()
        pg_hint = (
            pages[strong_pages[0]].page_id
            if len(strong_pages) == 1
            else f"{pages[strong_pages[0]].page_id}..{pages[strong_pages[-1]].page_id}"
        )
        snippet = s[:68] + ("..." if len(s) > 68 else "")
        pin_str = ""
        if sent_pins[i]:
            pin_ids = ", ".join(dict.fromkeys(pid for _, pid, _ in sent_pins[i]))
            pin_str = f" | BẮT BUỘC GẮN PANEL THOẠI: [{pin_ids}]"
        lines.append(f"- Câu {i + 1} (\"{snippet}\") → neo [{pg_hint}]{pin_str}")
    return "\n".join(lines)


def verify_beats_alignment(
    beats: list[ScriptBeat],
    pages: list[PageAlignmentRecord],
) -> list[AlignmentIssue]:
    """Verify every beat's assigned panels match its quoted dialogue and content events."""
    if len(pages) < 3 or not beats:
        return []

    index = _PartAlignmentIndex(pages)
    n_pg = len(pages)
    issues: list[AlignmentIssue] = []

    for bi, beat in enumerate(beats):
        if beat.beat_type == "hook_montage":
            continue
        nar = beat.narration_vi.strip()
        if not nar:
            continue
        assigned_pg_ids = list(
            dict.fromkeys(_panel_to_page_id(pid) for pid in beat.panel_ids if _panel_to_page_id(pid) in index.page_idx)
        )
        if not assigned_pg_ids:
            continue
        assigned_idxs = [index.page_idx[pg_id] for pg_id in assigned_pg_ids]

        # 1. Exact Quoted Dialogue Pin check (highest precision)
        quote_flagged_pages: set[int] = set()
        for q in _extract_quoted_spans(nar):
            pin = index.find_quote_pin(q)
            if not pin:
                continue
            _, q_pgi, q_pid = pin
            if q_pgi not in assigned_idxs:
                exp_pg = pages[q_pgi].page_id
                quote_flagged_pages.add(q_pgi)
                issues.append(
                    AlignmentIssue(
                        beat_idx=bi,
                        beat_id=beat.beat_id,
                        kind="quote_mismatch",
                        expected_page_id=exp_pg,
                        expected_panel_id=q_pid,
                        assigned_pages=assigned_pg_ids,
                        message=(
                            f"{beat.beat_id} trích thoại nằm ở {q_pid} ({exp_pg}) "
                            f"nhưng lại gắn trang {assigned_pg_ids} — phải gắn đúng {q_pid}."
                        ),
                    )
                )

        # 2. Content-to-Page Phase-Lag check (requires concrete lexical + bigram evidence)
        toks = _tokenize(nar)
        if len(toks) < 6:
            continue
        scored = [index.score_text_on_page(nar, j) for j in range(n_pg)]
        best_j = max(range(n_pg), key=lambda j: scored[j][0])
        best_sc, best_u, best_b = scored[best_j]
        if best_j in quote_flagged_pages:
            continue

        # Require concrete lexical specificity so short abstract commentary beats never false-positive
        has_concrete_match = (best_b >= 1 and best_u >= 3) or (best_u >= 4)
        if not has_concrete_match:
            continue

        tol_idxs = [j for j in range(n_pg) if min(abs(j - a) for a in assigned_idxs) <= 1]
        tol_best_sc = max((scored[j][0] for j in tol_idxs), default=0.0)
        dist = min(abs(best_j - a) for a in assigned_idxs)

        if dist >= 2 and best_sc >= 0.22 and best_sc >= tol_best_sc * 1.9 + 0.07:
            exp_pg = pages[best_j].page_id
            exp_pid = pages[best_j].panels[0][0] if pages[best_j].panels else None
            issues.append(
                AlignmentIssue(
                    beat_idx=bi,
                    beat_id=beat.beat_id,
                    kind="visual_desync",
                    expected_page_id=exp_pg,
                    expected_panel_id=exp_pid,
                    assigned_pages=assigned_pg_ids,
                    message=(
                        f"{beat.beat_id} kể nội dung của trang {exp_pg} (khớp vượt trội {best_sc:.2f} vs {tol_best_sc:.2f}) "
                        f"nhưng đang gắn lệch {dist} trang ở {assigned_pg_ids}."
                    ),
                )
            )
    return issues


def pin_missing_quote_panels(
    beats: list[ScriptBeat],
    pages: list[PageAlignmentRecord],
    allowed_panels: list[str],
) -> int:
    """Deterministic safety-net: ensure any quoted dialogue beat includes its exact quote panel without backward page jumps."""
    if not beats or not pages or not allowed_panels:
        return 0
    panel_rank = {pid: i for i, pid in enumerate(allowed_panels)}
    index = _PartAlignmentIndex(pages)
    pinned = 0

    for bi, beat in enumerate(beats):
        if beat.beat_type == "hook_montage":
            continue
        prev_max_pg = (
            max(
                (index.page_idx[_panel_to_page_id(p)] for p in beats[bi - 1].panel_ids if _panel_to_page_id(p) in index.page_idx),
                default=-1,
            )
            if bi > 0
            else -1
        )
        next_min_pg = (
            min(
                (index.page_idx[_panel_to_page_id(p)] for p in beats[bi + 1].panel_ids if _panel_to_page_id(p) in index.page_idx),
                default=len(pages),
            )
            if bi + 1 < len(beats)
            else len(pages)
        )
        cur_pages = {_panel_to_page_id(p) for p in beat.panel_ids}
        for q in _extract_quoted_spans(beat.narration_vi):
            pin = index.find_quote_pin(q)
            if not pin:
                continue
            _, q_pgi, q_pid = pin
            q_pg = pages[q_pgi].page_id
            if q_pg in cur_pages or q_pid not in panel_rank:
                continue
            if prev_max_pg <= q_pgi <= next_min_pg:
                new_pids = list(beat.panel_ids)
                if len(new_pids) >= 4:
                    new_pids[-1] = q_pid
                else:
                    new_pids.append(q_pid)
                beat.panel_ids = list(dict.fromkeys(sorted(new_pids, key=lambda p: panel_rank.get(p, 0))))
                cur_pages.add(q_pg)
                pinned += 1
    return pinned
