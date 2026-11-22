"""Stage 5 Agent: Frame Choreographer mapping audio beats to smart camera shots, diptychs & subtitles."""

from __future__ import annotations

import math
import re
from pathlib import Path

from src.config import RenderConfig
from src.schemas.ingestion import JobManifest, PanelInfo
from src.schemas.script import RecapScript
from src.schemas.timeline import (
    AudioManifest,
    SubtitleCue,
    TimelineShot,
    VideoTimeline,
)


def _split_caption_chunks(text: str, max_words: int = 8) -> list[str]:
    words = text.strip().split()
    if not words:
        return []
    chunks: list[str] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or (len(cur) >= 4 and re.search(r"[.,!?;:]$", w)):
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _is_diptych_candidate(p1: PanelInfo, p2: PanelInfo, page_short: int = 1000) -> bool:
    """Two consecutive distinct tall narrow panels on the same page pair cleanly side-by-side in 16:9."""
    if p1.panel_id == p2.panel_id or p1.page_id != p2.page_id:
        return False
    r1 = p1.width / max(1, p1.height)
    r2 = p2.width / max(1, p2.height)
    min_side1 = min(p1.width, p1.height)
    min_side2 = min(p2.width, p2.height)
    min_diptych_side = max(48, int(page_short * 0.14))
    return (
        0.18 <= r1 < 0.72
        and 0.18 <= r2 < 0.72
        and min_side1 >= min_diptych_side
        and min_side2 >= min_diptych_side
    )


def _is_good_visual_panel(p: PanelInfo, page_short: int = 1000) -> bool:
    """Filter out extreme slivers or empty background/screentone strips using page-relative size."""
    min_side = min(p.width, p.height)
    ar = p.width / max(1, p.height)
    has_face = any(fp.kind == "face" for fp in p.focal_points)
    has_text = any(fp.kind == "text" for fp in p.focal_points)
    min_panel_side = max(36, int(page_short * 0.10))
    if min_side < min_panel_side:
        return False
    if (ar < 0.18 or ar > 4.2) and not (has_face and has_text):
        return False
    if (ar < 0.25 or ar > 3.5) and not (has_face or has_text):
        return False
    return True


class ChoreographerAgent:
    """Allocates panel shots, diptych pairs, aspect-aware camera motions, transitions, and subtitles."""

    def __init__(self, render_config: RenderConfig) -> None:
        self.cfg = render_config

    def choreograph(
        self,
        job_id: str,
        manifest: JobManifest,
        script: RecapScript,
        audio_manifest: AudioManifest,
        job_dir: Path,
        force: bool = False,
    ) -> VideoTimeline:
        tl_path = job_dir / "05_timeline.json"
        if tl_path.exists() and not force:
            return VideoTimeline.model_validate_json(tl_path.read_text(encoding="utf-8"))

        all_panels: list[PanelInfo] = [
            p
            for ch in manifest.chapters
            for pg in ch.pages
            for p in pg.panels
        ]
        # Exclude skipped backmatter pages (if tree_index.json exists) so borrowing never leaks into extras
        skipped_pages: set[str] = set()
        tree_file = job_dir / "repo" / "tree_index.json"
        if tree_file.exists():
            try:
                import json

                skipped_pages = set(json.loads(tree_file.read_text(encoding="utf-8")).get("skipped_pages", []))
            except Exception:
                skipped_pages = set()

        panel_map: dict[str, PanelInfo] = {p.panel_id: p for p in all_panels}
        panel_pos: dict[str, int] = {p.panel_id: idx for idx, p in enumerate(all_panels)}
        page_short_map: dict[str, int] = {
            pg.page_id: max(1, min(pg.width, pg.height))
            for ch in manifest.chapters
            for pg in ch.pages
        }
        audio_by_beat = {b.beat_id: b for b in audio_manifest.beats}

        shots: list[TimelineShot] = []
        subtitles: list[SubtitleCue] = []
        sub_idx = 1
        max_shot_ms = self.cfg.max_shot_duration_ms
        used_panel_ids: set[str] = set()
        prev_section: str | None = None

        for b_idx, beat in enumerate(script.beats):
            ab = audio_by_beat.get(beat.beat_id)
            if not ab:
                continue
            beat_start = ab.start_ms
            beat_end = ab.end_ms + ab.pause_after_ms
            beat_span = max(300, beat_end - beat_start)

            # Subtitles split proportionally by character count
            chunks = _split_caption_chunks(beat.narration_vi)
            if chunks:
                total_chars = sum(max(1, len(c)) for c in chunks)
                cur_t = beat_start
                spoken_span = max(300, ab.end_ms - ab.start_ms)
                for c_idx, chunk in enumerate(chunks):
                    c_dur = (
                        (ab.end_ms - cur_t)
                        if c_idx == len(chunks) - 1
                        else int(spoken_span * len(chunk) / total_chars)
                    )
                    c_end = min(ab.end_ms, cur_t + max(200, c_dur))
                    subtitles.append(
                        SubtitleCue(
                            index=sub_idx,
                            start_ms=cur_t,
                            end_ms=c_end,
                            text=chunk,
                        )
                    )
                    sub_idx += 1
                    cur_t = c_end

            # Deduplicate repeated panel_ids inside the same beat
            raw_pids = [pid for pid in beat.panel_ids if pid in panel_map]
            pids = list(dict.fromkeys(raw_pids))
            # Drop low-quality empty slivers only if the beat still retains usable panels
            good_pids = [
                pid
                for pid in pids
                if _is_good_visual_panel(
                    panel_map[pid], page_short_map.get(panel_map[pid].page_id, 1000)
                )
            ]
            if good_pids:
                pids = good_pids
            # Avoid repeating the final panel of the preceding beat at the start of this beat
            if len(pids) > 1 and shots:
                prev_shot_pids = {shots[-1].panel_id, shots[-1].secondary_panel_id} - {None}
                dedup_cross = [pid for pid in pids if pid not in prev_shot_pids]
                if dedup_cross:
                    pids = dedup_cross
            if not pids and all_panels:
                pids = [all_panels[b_idx % len(all_panels)].panel_id]
            used_panel_ids.update(pids)

            # Borrow unused same-page panels only so visuals never drift into unnarrated pages
            beat_pages = {panel_map[pid].page_id for pid in pids if pid in panel_map}
            next_beat_pids = (
                set(script.beats[b_idx + 1].panel_ids)
                if b_idx + 1 < len(script.beats)
                else set()
            )
            needed_slots = min(5, max(len(pids), math.ceil(beat_span / max(1000, max_shot_ms))))
            if len(pids) < needed_slots and all_panels and pids[0] in panel_pos:
                anchor_pos = panel_pos[pids[0]]
                cursor = anchor_pos + 1
                while len(pids) < needed_slots and cursor < len(all_panels) and (cursor - anchor_pos) <= 12:
                    cand = all_panels[cursor]
                    if (
                        cand.page_id in beat_pages
                        and cand.panel_id not in pids
                        and cand.panel_id not in used_panel_ids
                        and cand.panel_id not in next_beat_pids
                        and cand.page_id not in skipped_pages
                        and _is_good_visual_panel(cand, page_short_map.get(cand.page_id, 1000))
                    ):
                        pids.append(cand.panel_id)
                        used_panel_ids.add(cand.panel_id)
                    cursor += 1
                pids.sort(key=lambda pid: panel_pos[pid])

            # Group consecutive tall narrow panels into diptych pairs when enabled
            groups: list[tuple[PanelInfo, PanelInfo | None]] = []
            i = 0
            while i < len(pids):
                p1 = panel_map[pids[i]]
                if (
                    self.cfg.enable_diptych
                    and i + 1 < len(pids)
                    and _is_diptych_candidate(
                        p1, panel_map[pids[i + 1]], page_short_map.get(p1.page_id, 1000)
                    )
                    and (beat_span / max(1, len(groups) + len(pids) - i - 1)) <= max_shot_ms * 1.25
                ):
                    groups.append((p1, panel_map[pids[i + 1]]))
                    i += 2
                else:
                    groups.append((p1, None))
                    i += 1

            # Allocate per-shot durations weighted proportionally by spoken sentence length
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?…])\s+", beat.narration_vi.strip())
                if s.strip()
            ]
            num_groups = max(1, len(groups))
            if num_groups > 1 and len(sentences) >= 2:
                weights: list[float] = []
                n_sent = len(sentences)
                for g_idx in range(num_groups):
                    s_lo = int(g_idx * n_sent / num_groups)
                    s_hi = max(s_lo + 1, int((g_idx + 1) * n_sent / num_groups))
                    seg_len = sum(max(8, len(sentences[j])) for j in range(s_lo, min(n_sent, s_hi)))
                    weights.append(float(seg_len))
                avg_w = sum(weights) / num_groups
                clamped = [min(1.55 * avg_w, max(0.55 * avg_w, w)) for w in weights]
                total_w = sum(clamped) or float(num_groups)
                slot_durations = [max(300, int(round(beat_span * w / total_w))) for w in clamped]
            else:
                slot_durations = [beat_span // num_groups] * num_groups

            cur_shot_ms = beat_start
            for s_idx, (panel, sec_panel) in enumerate(groups):
                s_start = cur_shot_ms
                s_end = beat_end if s_idx == num_groups - 1 else min(beat_end, s_start + slot_durations[s_idx])
                cur_shot_ms = s_end
                dur_sec = max(0.3, (s_end - s_start) / 1000.0)
                delta = min(
                    self.cfg.max_zoom_delta,
                    max(self.cfg.min_zoom_delta, self.cfg.zoom_speed_per_sec * dur_sec),
                )

                aspect = panel.width / max(1, panel.height)
                face_fp = next((fp for fp in panel.focal_points if fp.kind == "face"), None)

                if sec_panel is not None:
                    fx, fy = 0.5, 0.5
                    move = "diptych"
                    z0, z1 = 1.0, round(1.0 + min(0.08, delta), 3)
                elif aspect < 0.65:
                    fx, fy = 0.5, 0.5
                    move = "pan_down" if s_idx % 2 == 0 else "pan_up"
                    z0, z1 = 1.22, round(1.22 + min(0.05, delta * 0.4), 3)
                elif aspect > 1.55:
                    fx, fy = 0.5, 0.5
                    move = "pan_left"
                    z0, z1 = 1.18, round(1.18 + min(0.05, delta * 0.4), 3)
                elif face_fp:
                    fx, fy = face_fp.norm_cx, face_fp.norm_cy
                    move = "focus_face"
                    z0 = 1.02
                    z1 = round(z0 + delta + self.cfg.face_zoom_bonus, 3)
                elif beat.mood in ("action", "explosive") and s_idx % 2 == 0:
                    fx, fy = 0.5, 0.5
                    move = "shake_impact"
                    z0, z1 = 1.04, round(1.04 + delta, 3)
                else:
                    fx, fy = 0.5, 0.45
                    move = "zoom_in" if s_idx % 2 == 0 else "zoom_out"
                    if move == "zoom_in":
                        z0, z1 = 1.0, round(1.0 + delta, 3)
                    else:
                        z0, z1 = round(1.0 + delta, 3), 1.0

                transition = "cut"
                if self.cfg.enable_transitions and shots:
                    if s_idx == 0:
                        if prev_section is not None and beat.section != prev_section:
                            transition = "fade_black"
                        elif beat.mood in ("action", "shocking", "explosive", "horrific"):
                            transition = "flash_white"
                        else:
                            transition = "crossfade"

                shots.append(
                    TimelineShot(
                        shot_id=f"{beat.beat_id}_s{s_idx + 1}",
                        beat_id=beat.beat_id,
                        panel_id=panel.panel_id,
                        panel_path=panel.crop_path,
                        secondary_panel_id=sec_panel.panel_id if sec_panel else None,
                        secondary_panel_path=sec_panel.crop_path if sec_panel else None,
                        start_ms=s_start,
                        end_ms=s_end,
                        camera_move=move,
                        focus_x=fx,
                        focus_y=fy,
                        zoom_start=z0,
                        zoom_end=z1,
                        transition_in=transition,
                        mood=beat.mood,
                    )
                )
            prev_section = beat.section

        timeline = VideoTimeline(
            job_id=job_id,
            width=self.cfg.width,
            height=self.cfg.height,
            fps=self.cfg.fps,
            total_duration_ms=audio_manifest.total_duration_ms,
            voiceover_path=audio_manifest.voiceover_path,
            bgm_path=self.cfg.bgm_path,
            shots=shots,
            subtitles=subtitles,
        )
        tl_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
        return timeline
