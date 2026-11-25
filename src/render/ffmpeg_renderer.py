"""FFmpeg + OpenCV video renderer outputting H.264 + stereo AAC MP4 with transitions & optional BGM."""

from __future__ import annotations

from functools import lru_cache
import os
import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from src.cv.manga_detector import trim_white_border_with_offset
from src.render.compositor import _make_blurred_bg, render_shot_frame
from src.render.registry import RENDERER_REGISTRY
from src.render.subtitles import draw_subtitle_overlay, warm_subtitle_cache
from src.schemas.timeline import VideoTimeline


@lru_cache(maxsize=2)
def _detect_vaapi_device(ffmpeg_exe: str) -> str | None:
    """Return working VAAPI render node (e.g. /dev/dri/renderD128) if h264_vaapi succeeds."""
    dev = "/dev/dri/renderD128"
    if not Path(dev).exists():
        return None
    try:
        r = subprocess.run(
            [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-vaapi_device",
                dev,
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=0.1:r=10",
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "h264_vaapi",
                "-qp",
                "24",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=6,
        )
        return dev if r.returncode == 0 else None
    except Exception:
        return None


def _apply_transition(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    transition_in: str,
    t_in_shot_ms: int,
) -> np.ndarray:
    if transition_in == "crossfade" and prev_frame is not None and 0 <= t_in_shot_ms < 180:
        alpha = t_in_shot_ms / 180.0
        return cv2.addWeighted(prev_frame, 1.0 - alpha, frame, alpha, 0.0)
    if transition_in == "flash_white" and 0 <= t_in_shot_ms < 150:
        w_alpha = (1.0 - (t_in_shot_ms / 150.0)) * 0.75
        return cv2.convertScaleAbs(frame, alpha=1.0 - w_alpha, beta=245.0 * w_alpha)
    if transition_in == "fade_black" and 0 <= t_in_shot_ms < 220:
        gain = max(0.05, t_in_shot_ms / 220.0)
        return cv2.convertScaleAbs(frame, alpha=gain, beta=0.0)
    return frame


def _trim_excess_white_border(img: np.ndarray) -> np.ndarray:
    """Crop dominant empty white margins on sparse single-panel inset pages."""
    trimmed, _, _ = trim_white_border_with_offset(img)
    return trimmed


@RENDERER_REGISTRY.register("ffmpeg_opencv")
class FFmpegOpenCVRenderer:
    """Streams composited frames to FFmpeg and muxes stereo AAC audio (+ optional ducked BGM)."""

    name = "ffmpeg_opencv"

    def __init__(
        self,
        crf: int = 22,
        preset: str = "veryfast",
        audio_bitrate: str = "192k",
        audio_channels: int = 2,
        burn_subtitles: bool = True,
        margin_ratio: float = 0.06,
        bottom_safe_ratio: float = 0.14,
        bgm_volume: float = 0.14,
        bgm_ducking: bool = True,
        subtitle_font_ratio: float = 0.036,
        use_hw_accel: bool = True,
        music_library_dir: Path | str | None = Path("assets/music_library"),
        min_bgm_cue_sec: float = 28.0,
    ) -> None:
        self.crf = crf
        self.preset = preset
        self.audio_bitrate = audio_bitrate
        self.audio_channels = audio_channels
        self.burn_subtitles = burn_subtitles
        self.margin_ratio = margin_ratio
        self.bottom_safe_ratio = bottom_safe_ratio
        self.bgm_volume = bgm_volume
        self.bgm_ducking = bgm_ducking
        self.subtitle_font_ratio = subtitle_font_ratio
        self.use_hw_accel = use_hw_accel
        self.music_library_dir = Path(music_library_dir) if music_library_dir else None
        self.min_bgm_cue_sec = max(5.0, float(min_bgm_cue_sec))

    def render(
        self,
        timeline: VideoTimeline,
        output_path: Path,
        subtitles_path: Path | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        vaapi_dev = _detect_vaapi_device(ffmpeg) if self.use_hw_accel else None

        total_frames = max(1, int(round(timeline.total_duration_ms / 1000.0 * timeline.fps)))
        has_audio = bool(timeline.voiceover_path and Path(timeline.voiceover_path).exists())
        active_bgm_path: str | None = timeline.bgm_path
        lib_has_catalog = bool(
            self.music_library_dir and (self.music_library_dir / "catalog.json").exists()
        )
        if has_audio and (
            (timeline.bgm_path and Path(timeline.bgm_path).name == "default_dark_fantasy.wav")
            or (timeline.bgm_path is None and lib_has_catalog)
        ):
            import json
            from src.audio.bgm_generator import build_dynamic_timeline_bgm

            tone_summary = ""
            series_title = ""
            bible_p = output_path.parent / "02b_story_bible.json"
            if bible_p.exists():
                try:
                    bdata = json.loads(bible_p.read_text(encoding="utf-8"))
                    tone_summary = str(bdata.get("tone_summary") or "")
                    series_title = str(bdata.get("series_title_guess") or "")
                except Exception:
                    pass

            series_music_p = output_path.parent / "audio" / "series_music_context.json"
            genre_prefs_ov: dict[str, float] | None = None
            avoid_init_tracks: list[str] | None = None
            if series_music_p.exists():
                try:
                    sm_data = json.loads(series_music_p.read_text(encoding="utf-8"))
                    if isinstance(sm_data.get("genre_preferences"), dict) and sm_data["genre_preferences"]:
                        genre_prefs_ov = {str(k): float(v) for k, v in sm_data["genre_preferences"].items()}
                    if isinstance(sm_data.get("avoid_initial_track_ids"), list):
                        avoid_init_tracks = [str(x) for x in sm_data["avoid_initial_track_ids"] if x]
                except Exception:
                    pass

            dyn_bgm = output_path.parent / "audio" / "04b_bgm_dynamic.wav"
            custom_dir = Path(timeline.bgm_path).parent if timeline.bgm_path else None
            build_dynamic_timeline_bgm(
                timeline,
                dyn_bgm,
                custom_bgm_dir=custom_dir,
                music_library_dir=self.music_library_dir,
                tone_summary=tone_summary,
                title=series_title,
                min_cue_duration_ms=int(round(self.min_bgm_cue_sec * 1000)),
                credits_path=output_path.parent / "exports" / "music_credits.txt",
                cues_json_path=output_path.parent / "audio" / "04b_music_cues.json",
                genre_prefs_override=genre_prefs_ov,
                avoid_initial_track_ids=avoid_init_tracks,
            )
            active_bgm_path = str(dyn_bgm.resolve())
        elif timeline.bgm_path:
            bgm_file = Path(timeline.bgm_path)
            if not bgm_file.exists() and bgm_file.suffix.lower() == ".wav":
                from src.audio.bgm_generator import ensure_default_bgm

                ensure_default_bgm(bgm_file)
        has_bgm = bool(has_audio and active_bgm_path and Path(active_bgm_path).exists())

        cmd = [ffmpeg, "-y"]
        if vaapi_dev:
            cmd.extend(["-vaapi_device", vaapi_dev])
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{timeline.width}x{timeline.height}",
                "-r",
                str(timeline.fps),
                "-i",
                "-",
            ]
        )
        if has_audio:
            cmd.extend(["-i", str(timeline.voiceover_path)])
        if has_bgm:
            cmd.extend(["-stream_loop", "-1", "-i", str(active_bgm_path)])

        if has_bgm:
            vol = max(0.02, min(0.6, self.bgm_volume))
            if self.bgm_ducking:
                afc = (
                    f"[1:a]asplit=2[vo_sc][vo_mix];"
                    f"[2:a]volume={vol:.3f}[bgm];"
                    f"[bgm][vo_sc]sidechaincompress=threshold=0.02:ratio=6:attack=25:release=350[bgm_duck];"
                    f"[vo_mix][bgm_duck]amix=inputs=2:duration=first:normalize=0[aout]"
                )
            else:
                afc = f"[2:a]volume={vol:.3f}[bgm];[1:a][bgm]amix=inputs=2:duration=first:normalize=0[aout]"

            if vaapi_dev:
                fc = f"[0:v]format=nv12,hwupload[vout];{afc}"
                cmd.extend(
                    [
                        "-filter_complex",
                        fc,
                        "-map",
                        "[vout]",
                        "-map",
                        "[aout]",
                        "-c:v",
                        "h264_vaapi",
                        "-qp",
                        str(self.crf),
                    ]
                )
            else:
                cmd.extend(
                    [
                        "-filter_complex",
                        afc,
                        "-map",
                        "0:v:0",
                        "-map",
                        "[aout]",
                        "-c:v",
                        "libx264",
                        "-preset",
                        self.preset,
                        "-crf",
                        str(self.crf),
                        "-threads",
                        "0",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                )
            cmd.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    self.audio_bitrate,
                    "-ac",
                    str(self.audio_channels),
                    "-ar",
                    "48000",
                    "-shortest",
                ]
            )
        else:
            if vaapi_dev:
                cmd.extend(
                    [
                        "-vf",
                        "format=nv12,hwupload",
                        "-c:v",
                        "h264_vaapi",
                        "-qp",
                        str(self.crf),
                    ]
                )
            else:
                cmd.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-preset",
                        self.preset,
                        "-crf",
                        str(self.crf),
                        "-threads",
                        "0",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                )
            if has_audio:
                cmd.extend(
                    [
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:a",
                        "aac",
                        "-b:a",
                        self.audio_bitrate,
                        "-ac",
                        str(self.audio_channels),
                        "-ar",
                        "48000",
                        "-shortest",
                    ]
                )
        cmd.extend(["-movflags", "+faststart", str(output_path)])

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None

        panel_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        bg_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        max_cache_items = 16
        max_panel_dim = int(max(timeline.width, timeline.height) * 1.6)
        shots = timeline.shots
        subs = timeline.subtitles

        if self.burn_subtitles and subs:
            warm_subtitle_cache(
                [s.text for s in subs],
                timeline.width,
                timeline.height,
                font_ratio=self.subtitle_font_ratio,
            )

        def _load_img(pid: str, ppath: str) -> np.ndarray:
            cached = panel_cache.get(pid)
            if cached is not None:
                panel_cache.move_to_end(pid)
                return cached
            img = cv2.imread(ppath)
            if img is None:
                img = np.full((400, 400, 3), 30, dtype=np.uint8)
            else:
                img = _trim_excess_white_border(img)
                ih, iw = img.shape[:2]
                longest = max(ih, iw)
                if longest > max_panel_dim:
                    sc = max_panel_dim / float(longest)
                    img = cv2.resize(
                        img,
                        (max(2, int(round(iw * sc))), max(2, int(round(ih * sc)))),
                        interpolation=cv2.INTER_AREA,
                    )
            panel_cache[pid] = img
            if len(panel_cache) > max_cache_items:
                panel_cache.popitem(last=False)
            return img

        # Pre-assign every frame index to its (shot_idx, t_ms, sub_text) once in O(N)
        frame_plan: list[tuple[int, int, str]] = []
        s_ptr = 0
        sub_ptr = 0
        for f_idx in range(total_frames):
            t_ms = int(round(f_idx * 1000.0 / timeline.fps))
            while s_ptr + 1 < len(shots) and t_ms >= shots[s_ptr].end_ms:
                s_ptr += 1
            sub_txt = ""
            if self.burn_subtitles and subs:
                while sub_ptr + 1 < len(subs) and t_ms > subs[sub_ptr].end_ms:
                    sub_ptr += 1
                cur_sub = subs[sub_ptr]
                if cur_sub.start_ms <= t_ms <= cur_sub.end_ms:
                    sub_txt = cur_sub.text
            frame_plan.append((s_ptr if shots else -1, t_ms, sub_txt))

        # Group contiguous frames by shot_idx for lock-free multi-threaded compositing
        shot_runs: list[tuple[int, list[tuple[int, str]]]] = []
        for s_idx, t_ms, sub_txt in frame_plan:
            if not shot_runs or shot_runs[-1][0] != s_idx:
                shot_runs.append((s_idx, [(t_ms, sub_txt)]))
            else:
                shot_runs[-1][1].append((t_ms, sub_txt))

        num_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        prev_shot_last_frame: np.ndarray | None = None
        black_frame = np.zeros((timeline.height, timeline.width, 3), dtype=np.uint8)

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for run_idx, (s_idx, items) in enumerate(shot_runs):
                    if s_idx < 0:
                        for _, sub_txt in items:
                            frame = (
                                draw_subtitle_overlay(
                                    black_frame, sub_txt, font_ratio=self.subtitle_font_ratio
                                )
                                if sub_txt
                                else black_frame
                            )
                            proc.stdin.write(memoryview(frame))
                        continue

                    shot = shots[s_idx]
                    primary = _load_img(shot.panel_id, shot.panel_path)
                    secondary = (
                        _load_img(shot.secondary_panel_id, shot.secondary_panel_path)
                        if shot.secondary_panel_id and shot.secondary_panel_path
                        else None
                    )
                    # Build blurred background once on main thread
                    if shot.panel_id not in bg_cache:
                        bg_cache[shot.panel_id] = _make_blurred_bg(
                            primary, timeline.width, timeline.height
                        )
                        if len(bg_cache) > max_cache_items:
                            bg_cache.popitem(last=False)
                    else:
                        bg_cache.move_to_end(shot.panel_id)
                    local_bg = {shot.panel_id: bg_cache[shot.panel_id]}

                    span = max(1, shot.end_ms - shot.start_ms)
                    prev_frame_ref = prev_shot_last_frame

                    def _compose_one(item: tuple[int, str]) -> np.ndarray:
                        t_ms, sub_txt = item
                        prog = (t_ms - shot.start_ms) / span
                        raw = render_shot_frame(
                            primary,
                            shot,
                            prog,
                            timeline.width,
                            timeline.height,
                            bg_cache=local_bg,
                            secondary_bgr=secondary,
                            margin_ratio=self.margin_ratio,
                            bottom_safe_ratio=self.bottom_safe_ratio,
                            reading_order=timeline.reading_order,
                        )
                        out_f = _apply_transition(
                            raw,
                            prev_frame_ref,
                            shot.transition_in,
                            t_ms - shot.start_ms,
                        )
                        if sub_txt:
                            out_f = draw_subtitle_overlay(
                                out_f,
                                sub_txt,
                                in_place=True,
                                font_ratio=self.subtitle_font_ratio,
                            )
                        return np.ascontiguousarray(out_f)

                    # Pre-compute end-of-shot raw frame only if next shot crossfades into it
                    next_needs_prev = (
                        run_idx + 1 < len(shot_runs)
                        and shot_runs[run_idx + 1][0] >= 0
                        and shots[shot_runs[run_idx + 1][0]].transition_in == "crossfade"
                    )
                    if next_needs_prev:
                        last_t_ms = items[-1][0]
                        prev_shot_last_frame = render_shot_frame(
                            primary,
                            shot,
                            (last_t_ms - shot.start_ms) / span,
                            timeline.width,
                            timeline.height,
                            bg_cache=local_bg,
                            secondary_bgr=secondary,
                            margin_ratio=self.margin_ratio,
                            bottom_safe_ratio=self.bottom_safe_ratio,
                            reading_order=timeline.reading_order,
                        )
                    else:
                        prev_shot_last_frame = None

                    if num_workers > 1 and len(items) >= 8:
                        for frame in pool.map(_compose_one, items):
                            proc.stdin.write(memoryview(frame))
                    else:
                        for item in items:
                            frame = _compose_one(item)
                            proc.stdin.write(memoryview(frame))
        finally:
            proc.stdin.close()

        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(
                f"FFmpeg render failed (rc={ret}): {stderr_bytes.decode('utf-8', errors='replace')[-600:]}"
            )
        return output_path
