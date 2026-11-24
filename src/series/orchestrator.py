"""Multi-Volume Series Batch Orchestrator: auto-discovers volumes, runs sequentially with continuity & optional compilation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from src.config import AppConfig
from src.cv.loader import IMAGE_EXTS, natural_sort_key
from src.pipeline import MangaRecapPipeline
from src.series.compiler import CompilationResult, compile_series_videos
from src.series.memory import SeriesStore

ARCHIVE_EXTENSIONS = {".cbz", ".zip"}


class SeriesRunReport(BaseModel):
    series_id: str
    series_dir: str
    volumes_processed: list[int] = Field(default_factory=list)
    volume_outputs: dict[int, str] = Field(default_factory=dict)
    compilations: list[CompilationResult] = Field(default_factory=list)


def discover_series_volumes(input_dir: Path) -> list[tuple[int, Path]]:
    """Discover ordered volume inputs (.cbz/.zip archives or volume subdirectories) via natural_sort_key."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Series input path does not exist: {root}")
    if root.is_file():
        return [(1, root)]

    archives = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ARCHIVE_EXTENSIONS],
        key=lambda p: natural_sort_key(p.name),
    )
    if archives:
        return [(idx, p) for idx, p in enumerate(archives, start=1)]

    # Otherwise check immediate subdirectories containing images, chapter subdirs, or archives
    subdirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: natural_sort_key(p.name),
    )
    vol_dirs: list[Path] = []
    for sd in subdirs:
        has_media = any(
            (f.is_file() and (f.suffix.lower() in IMAGE_EXTS or f.suffix.lower() in ARCHIVE_EXTENSIONS))
            for f in sd.rglob("*")
        )
        if has_media:
            vol_dirs.append(sd)

    if vol_dirs:
        return [(idx, p) for idx, p in enumerate(vol_dirs, start=1)]

    raise FileNotFoundError(f"No manga volume archives (.cbz/.zip) or volume folders found in {root}")


class SeriesOrchestrator:
    """Runs an ordered multi-volume series sequentially with persistent Series Memory & Arc/Full compilation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pipeline = MangaRecapPipeline(config)

    def run_series(
        self,
        input_dir: Path,
        series_id: str,
        continuity_mode: str = "episodic",
        persona: str | None = None,
        from_vol: int = 1,
        to_vol: int | None = None,
        compile_every: int | None = None,
        compile_all: bool = False,
        force: bool = False,
        skip_render: bool = False,
        on_volume_start: Callable[[int, int, Path, str], None] | None = None,
    ) -> SeriesRunReport:
        discovered = discover_series_volumes(input_dir)
        total_vols = len(discovered)
        hi = to_vol if to_vol is not None else total_vols
        selected = [(v_idx, v_path) for v_idx, v_path in discovered if from_vol <= v_idx <= hi]

        series_dir = self.config.project.series_dir / series_id
        store = SeriesStore(series_dir, series_id)
        chosen_persona = persona or self.config.project.default_persona

        report = SeriesRunReport(series_id=series_id, series_dir=str(series_dir.resolve()))
        prev_job_dir: Path | None = None
        arc_buffer: list[int] = []

        for v_idx, v_path in selected:
            job_id = f"{series_id}_v{v_idx:02d}"
            # Reload SeriesContext before EVERY volume so manual edits in glossary.yaml take effect immediately
            ctx = store.build_context(
                volume_index=v_idx,
                total_volumes=total_vols,
                continuity_mode=continuity_mode,
                persona=chosen_persona,
                tts_voice=self.config.tts.voice,
            )
            if on_volume_start is not None:
                on_volume_start(v_idx, total_vols, v_path, job_id)

            out_path = self.pipeline.run(
                input_path=v_path,
                job_id=job_id,
                persona=persona,
                force=force,
                skip_render=skip_render,
                prev_job_dir=prev_job_dir,
                series_ctx=ctx,
            )
            report.volumes_processed.append(v_idx)
            report.volume_outputs[v_idx] = str(out_path)
            prev_job_dir = self.config.project.jobs_dir / job_id
            arc_buffer.append(v_idx)

            if compile_every and compile_every > 1 and not skip_render and len(arc_buffer) >= compile_every:
                state, _ = store.load_or_init(persona=chosen_persona, tts_voice=self.config.tts.voice)
                comp = compile_series_videos(
                    state=state,
                    jobs_dir=self.config.project.jobs_dir,
                    out_dir=store.compilations_dir,
                    volumes=list(arc_buffer),
                )
                report.compilations.append(comp)
                arc_buffer.clear()

        # Flush remaining partial arc group if compile_every was requested
        if compile_every and compile_every > 1 and not skip_render and len(arc_buffer) >= 2:
            state, _ = store.load_or_init(persona=chosen_persona, tts_voice=self.config.tts.voice)
            comp = compile_series_videos(
                state=state,
                jobs_dir=self.config.project.jobs_dir,
                out_dir=store.compilations_dir,
                volumes=list(arc_buffer),
            )
            report.compilations.append(comp)

        if compile_all and not skip_render and len(report.volumes_processed) >= 1:
            state, _ = store.load_or_init(persona=chosen_persona, tts_voice=self.config.tts.voice)
            comp = compile_series_videos(
                state=state,
                jobs_dir=self.config.project.jobs_dir,
                out_dir=store.compilations_dir,
                volumes=[e.volume_index for e in state.ledger],
                output_name=f"{series_id}_full_series",
            )
            report.compilations.append(comp)

        return report
