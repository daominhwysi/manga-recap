"""Arc & Full-Series Video Compiler: merges volume MP4s, shifts SRT timestamps, generates YouTube chapters & credits."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
from pydantic import BaseModel, Field

from src.schemas.series import SeriesState


class CompilationResult(BaseModel):
    output_mp4: str
    output_srt: str = ""
    chapters_path: str = ""
    credits_path: str = ""
    volumes_included: list[int] = Field(default_factory=list)
    total_duration_sec: float = 0.0


def parse_volume_range(spec: str | None, available_vols: list[int]) -> list[int]:
    """Parse specs like '1-6', '1,3,5', '1-3,7-9', or None (= all available)."""
    avail_sorted = sorted(set(available_vols))
    if not spec or not spec.strip():
        return avail_sorted
    wanted: set[int] = set()
    for token in re.split(r"[,;\s]+", spec.strip()):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                lo, hi = int(parts[0]), int(parts[1])
                if lo <= hi:
                    wanted.update(range(lo, hi + 1))
            except ValueError:
                continue
        else:
            try:
                wanted.add(int(token))
            except ValueError:
                continue
    return [v for v in avail_sorted if v in wanted]


def _parse_srt_ts_ms(ts: str) -> int:
    m = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", ts.strip())
    if not m:
        return 0
    h, mi, s, ms = (int(m.group(i)) for i in range(1, 5))
    return ((h * 3600 + mi * 60 + s) * 1000) + ms


def _format_srt_ts_ms(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    mi, rem = divmod(rem, 60_000)
    s, rem = divmod(rem, 1000)
    return f"{h:02d}:{mi:02d}:{s:02d},{rem:03d}"


def _format_yt_timestamp(ms: int, include_hours: bool = False) -> str:
    sec = max(0, int(round(ms / 1000.0)))
    h, rem = divmod(sec, 3600)
    mi, s = divmod(rem, 60)
    if include_hours or h > 0:
        return f"{h:02d}:{mi:02d}:{s:02d}"
    return f"{mi:02d}:{s:02d}"


def merge_srt_files(
    srt_entries: list[tuple[Path, int]],
    out_srt: Path,
) -> Path:
    """Merge multiple SRT files with exact cumulative millisecond offsets and renumber 1..N."""
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    merged_blocks: list[str] = []
    cue_num = 1
    ts_pat = re.compile(
        r"^(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})(.*)$"
    )
    for srt_path, offset_ms in srt_entries:
        if not srt_path.exists():
            continue
        raw = srt_path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        for block in re.split(r"\r?\n\r?\n+", raw):
            lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue
            ts_idx = 1 if lines[0].isdigit() else 0
            if ts_idx >= len(lines):
                continue
            m = ts_pat.match(lines[ts_idx])
            if not m:
                continue
            s_ms = _parse_srt_ts_ms(m.group(1)) + offset_ms
            e_ms = _parse_srt_ts_ms(m.group(2)) + offset_ms
            text_lines = lines[ts_idx + 1 :]
            if not text_lines:
                continue
            merged_blocks.append(
                f"{cue_num}\n{_format_srt_ts_ms(s_ms)} --> {_format_srt_ts_ms(e_ms)}\n"
                + "\n".join(text_lines)
            )
            cue_num += 1

    out_srt.write_text("\n\n".join(merged_blocks) + ("\n" if merged_blocks else ""), encoding="utf-8")
    return out_srt


def write_youtube_chapters(
    chapters: list[tuple[int, str]],
    out_path: Path,
    total_duration_ms: int = 0,
) -> Path:
    """Write YouTube chapter timestamp markers (00:00 Tập 1: ...)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_hours = total_duration_ms >= 3_600_000 or any(ms >= 3_600_000 for ms, _ in chapters)
    lines = [
        f"{_format_yt_timestamp(ms, include_hours=use_hours)} {label}"
        for ms, label in chapters
    ]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


def merge_music_credits(credit_paths: list[Path], out_path: Path) -> Path:
    """Deduplicate CC-BY music attribution lines across all merged volumes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines = ["# Music Credits (Merged Series Compilation)", ""]
    for cp in credit_paths:
        if not cp.exists():
            continue
        for raw in cp.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if s.startswith("- ") and s not in seen:
                seen.add(s)
                lines.append(s)
    if len(lines) == 2:
        lines.append("- Built-in procedural ambient score (Royalty-Free)")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def probe_media_duration_ms(mp4_path: Path, fallback_job_dir: Path | None = None) -> int:
    """Read accurate duration in ms via FFmpeg stderr probe (fallback to 05_timeline.json)."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(mp4_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2})\.(\d{2})", proc.stderr or "")
        if m:
            h, mi, s, cs = (int(m.group(i)) for i in range(1, 5))
            return ((h * 3600 + mi * 60 + s) * 1000) + cs * 10
    except Exception:
        pass

    if fallback_job_dir is not None:
        tl = fallback_job_dir / "05_timeline.json"
        if tl.exists():
            try:
                data = json.loads(tl.read_text(encoding="utf-8"))
                return int(data.get("total_duration_ms", 0) or 0)
            except Exception:
                pass
    return 0


def compile_series_videos(
    state: SeriesState,
    jobs_dir: Path,
    out_dir: Path,
    volumes: list[int] | None = None,
    output_name: str | None = None,
    reencode: bool = False,
) -> CompilationResult:
    """Concatenate rendered volume MP4s into a single Arc/Full-Series video + merged SRT, chapters & credits."""
    entries_by_vol = {e.volume_index: e for e in state.ledger}
    selected_vols = sorted(volumes) if volumes is not None else sorted(entries_by_vol.keys())

    valid_items: list[tuple[int, str, Path, Path, int]] = []
    for v in selected_vols:
        entry = entries_by_vol.get(v)
        if entry is None:
            continue
        job_dir = jobs_dir / entry.job_id
        mp4 = Path(entry.output_mp4) if entry.output_mp4 else (job_dir / "06_output.mp4")
        if not mp4.exists():
            mp4 = job_dir / "06_output.mp4"
        if not mp4.exists():
            continue
        dur_ms = probe_media_duration_ms(mp4, fallback_job_dir=job_dir)
        if dur_ms <= 0 and entry.duration_sec > 0:
            dur_ms = int(round(entry.duration_sec * 1000))
        title = entry.title_vi or f"Tập {v}"
        valid_items.append((v, title, job_dir, mp4.resolve(), dur_ms))

    if not valid_items:
        raise FileNotFoundError(
            f"No rendered MP4 files found for series '{state.series_id}' (requested volumes={selected_vols})."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    v_lo, v_hi = valid_items[0][0], valid_items[-1][0]
    stem = output_name or f"{state.series_id}_vol{v_lo:02d}_to_vol{v_hi:02d}"
    out_mp4 = out_dir / f"{stem}.mp4"
    out_srt = out_dir / f"{stem}.srt"
    out_chapters = out_dir / f"{stem}_chapters.txt"
    out_credits = out_dir / f"{stem}_music_credits.txt"

    # Build FFmpeg concat list
    concat_list = out_dir / f".{stem}_concat.txt"
    concat_lines = [
        f"file '{str(mp4).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
        for _, _, _, mp4, _ in valid_items
    ]
    concat_list.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        if not reencode:
            cmd_copy = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ]
            res = subprocess.run(cmd_copy, capture_output=True, text=True)
            if res.returncode != 0:
                reencode = True

        if reencode:
            cmd_enc = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ac",
                "2",
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ]
            res = subprocess.run(cmd_enc, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg series compilation failed: {res.stderr[-600:]}")
    finally:
        if concat_list.exists():
            concat_list.unlink()

    # Merge SRTs, YouTube Chapters, and Music Credits with exact cumulative timing
    srt_entries: list[tuple[Path, int]] = []
    chapters: list[tuple[int, str]] = []
    credit_paths: list[Path] = []
    cum_ms = 0
    for v, title, job_dir, _, dur_ms in valid_items:
        chapters.append((cum_ms, f"Tập {v}: {title}"))
        srt_p = job_dir / "06_subtitles.srt"
        if srt_p.exists():
            srt_entries.append((srt_p, cum_ms))
        cred_p = job_dir / "exports" / "music_credits.txt"
        if cred_p.exists():
            credit_paths.append(cred_p)
        cum_ms += dur_ms

    merge_srt_files(srt_entries, out_srt)
    write_youtube_chapters(chapters, out_chapters, total_duration_ms=cum_ms)
    merge_music_credits(credit_paths, out_credits)

    return CompilationResult(
        output_mp4=str(out_mp4.resolve()),
        output_srt=str(out_srt.resolve()),
        chapters_path=str(out_chapters.resolve()),
        credits_path=str(out_credits.resolve()),
        volumes_included=[v for v, _, _, _, _ in valid_items],
        total_duration_sec=round(cum_ms / 1000.0, 2),
    )
