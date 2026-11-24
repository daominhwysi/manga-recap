"""Typer CLI entrypoint for manga-recap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.downloaders.nyaa import NyaaDownloader
from src.pipeline import MangaRecapPipeline

app = typer.Typer(help="Automated multi-agent Manga & Manhwa Recap Video Generator")
music_app = typer.Typer(help="Manage automated royalty-free BGM & SFX music dataset")
series_app = typer.Typer(help="Manage multi-volume Series continuity, glossary, batch runs & Arc compilations")
app.add_typer(music_app, name="music")
app.add_typer(series_app, name="series")
console = Console()


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Path to manga folder, multi-chapter folder, or CBZ file"),
    job_id: str = typer.Option("demo_job", "--job-id", "-j", help="Output job identifier under jobs/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
    persona: Optional[str] = typer.Option(None, "--persona", "-p", help="Narration persona"),
    tts_backend: Optional[str] = typer.Option(None, "--tts", help="Override TTS backend (edge_tts, vieneu)"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="Limit pages per chapter"),
    prev_job: Optional[Path] = typer.Option(None, "--prev-job", help="Optional previous volume job dir for cast continuity"),
    series_id: Optional[str] = typer.Option(None, "--series-id", "-s", help="Series identifier for persistent multi-volume memory"),
    volume: int = typer.Option(1, "--vol", help="Volume index (1-based) when running with --series-id"),
    continuity_mode: str = typer.Option("episodic", "--mode", help="Series continuity mode: episodic | seamless | standalone"),
    force: bool = typer.Option(False, "--force", "-f", help="Ignore cached checkpoints"),
    skip_render: bool = typer.Option(False, "--skip-render", help="Run stages 1-5 + exporters without MP4 render"),
) -> None:
    """Run the full 6-stage manga recap pipeline."""
    cfg = load_config(config_path)
    if tts_backend:
        cfg.tts.backend = tts_backend
    if max_pages is not None:
        cfg.cv.max_pages_per_chapter = max_pages

    series_ctx = None
    if series_id:
        from src.series.memory import SeriesStore

        store = SeriesStore(cfg.project.series_dir / series_id, series_id)
        series_ctx = store.build_context(
            volume_index=volume,
            continuity_mode=continuity_mode,
            persona=persona or cfg.project.default_persona,
            tts_voice=cfg.tts.voice,
        )

    console.print(
        f"[bold cyan]Starting job:[/] {job_id} (model={cfg.llm.default_model}"
        + (f", series={series_id} vol={volume} mode={continuity_mode}" if series_id else "")
        + ")"
    )
    pipeline = MangaRecapPipeline(cfg)
    result = pipeline.run(
        input_path=input_path,
        job_id=job_id,
        persona=persona,
        force=force,
        skip_render=skip_render,
        prev_job_dir=prev_job,
        series_ctx=series_ctx,
    )
    console.print(f"[bold green]Done![/] Output: {result}")


@app.command()
def inspect(
    job_dir: Path = typer.Argument(..., help="Path to a job directory under jobs/"),
) -> None:
    """Inspect telemetry, LLM tokens, script metrics, and Critic report for a job."""
    if not job_dir.exists():
        console.print(f"[bold red]Not found:[/] {job_dir}")
        raise typer.Exit(code=1)

    tel_path = job_dir / "telemetry.json"
    if tel_path.exists():
        tel = json.loads(tel_path.read_text(encoding="utf-8"))
        console.print(
            f"[bold cyan]Job:[/] {tel.get('job_id')} | default={tel.get('config_default_model')} | "
            f"vision={tel.get('config_vision_model')} | total={tel.get('total_duration_sec', 0):.2f}s"
        )
        t_stage = Table(title="Pipeline Stages")
        t_stage.add_column("Stage")
        t_stage.add_column("Duration (s)", justify="right")
        t_stage.add_column("Details")
        for st in tel.get("stages", []):
            t_stage.add_row(st.get("stage", ""), str(st.get("duration_sec", 0)), json.dumps(st.get("details", {}), ensure_ascii=False))
        console.print(t_stage)

        calls = tel.get("llm_calls", [])
        if calls:
            by_agent: dict[str, list[int]] = {}
            for c in calls:
                ag = c.get("agent", "unknown")
                agg = by_agent.setdefault(ag, [0, 0, 0])
                agg[0] += 1
                agg[1] += int(c.get("prompt_tokens", 0) or 0)
                agg[2] += int(c.get("completion_tokens", 0) or 0)
            t_tok = Table(title="LLM Token Usage")
            t_tok.add_column("Agent")
            t_tok.add_column("Calls", justify="right")
            t_tok.add_column("Prompt", justify="right")
            t_tok.add_column("Completion", justify="right")
            t_tok.add_column("Total", justify="right")
            tot_p = tot_c = 0
            for ag, (cnt, p_tok, c_tok) in by_agent.items():
                tot_p += p_tok
                tot_c += c_tok
                t_tok.add_row(ag, str(cnt), str(p_tok), str(c_tok), str(p_tok + c_tok))
            t_tok.add_row("[bold]TOTAL[/]", str(len(calls)), str(tot_p), str(tot_c), str(tot_p + tot_c))
            console.print(t_tok)

    script_path = job_dir / "03b_script_final.json"
    if script_path.exists():
        sc = json.loads(script_path.read_text(encoding="utf-8"))
        beats = sc.get("beats", [])
        wc = sum(len(b.get("narration_vi", "").split()) for b in beats)
        console.print(
            f"[bold green]Script:[/] hook={sc.get('hook_strategy', '?')} | beats={len(beats)} | words={wc}"
        )

    rep_path = job_dir / "03b_critic_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        console.print(
            f"[bold yellow]Critic:[/] passed={rep.get('passed')} | score={rep.get('score')} | issues={len(rep.get('issues', []))}"
        )


@app.command()
def download(
    source: str = typer.Argument(..., help="Nyaa.si URL (https://nyaa.si/view/1246645) or ID"),
    volume: Optional[str] = typer.Option(None, "--volume", "-v", help="Volume filter (e.g. v01, vol1)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Download directory"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
    no_extract: bool = typer.Option(False, "--no-extract", help="Skip extracting CBZ archives"),
) -> None:
    """Download manga volume(s) from Nyaa.si torrent."""
    cfg = load_config(config_path)
    out = output_dir or (cfg.project.downloads_dir / "nyaa")
    dl = NyaaDownloader(
        base_url=cfg.downloader.nyaa_base_url,
        timeout_sec=cfg.downloader.timeout_sec,
    )
    console.print(f"[bold cyan]Downloading from Nyaa:[/] {source} (volume={volume or 'all'})")
    res_path = dl.download(source, out, volume_filter=volume, extract=not no_extract)
    console.print(f"[bold green]Downloaded & ready at:[/] {res_path}")


@music_app.command("fetch")
def music_fetch(
    per_mood: int = typer.Option(3, "--per-mood", "-n", help="Number of starter CC-BY tracks to fetch per canonical mood"),
    library_dir: Optional[Path] = typer.Option(None, "--library-dir", "-d", help="Music library root directory"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Download a curated CC-BY 4.0 starter BGM dataset + SFX pack and auto-index catalog.json."""
    from src.audio.music_library import fetch_starter_music_library

    cfg = load_config(config_path)
    lib_dir = library_dir or cfg.render.music_library_dir
    console.print(f"[bold cyan]Fetching starter CC-BY music dataset[/] ({per_mood} tracks/mood) -> {lib_dir} ...")
    cat = fetch_starter_music_library(lib_dir, per_mood=per_mood)
    console.print(
        f"[bold green]Ready![/] Indexed {len(cat.tracks)} BGM tracks & {len(cat.sfx)} SFX clips at {lib_dir / 'catalog.json'}"
    )


@music_app.command("index")
def music_index(
    import_dir: Optional[Path] = typer.Argument(None, help="Optional folder of custom MP3/WAV tracks to import & auto-tag"),
    library_dir: Optional[Path] = typer.Option(None, "--library-dir", "-d", help="Music library root directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run DSP acoustic analysis on all tracks"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Analyze DSP acoustics (LUFS, BPM, brightness, intro-skip) and auto-tag BGM/SFX into catalog.json."""
    from src.audio.music_library import index_music_library

    cfg = load_config(config_path)
    lib_dir = library_dir or cfg.render.music_library_dir
    console.print(f"[bold cyan]Indexing music library:[/] {lib_dir}" + (f" (importing {import_dir})" if import_dir else ""))
    cat = index_music_library(lib_dir, import_dir=import_dir, force_reanalyze=force)
    console.print(
        f"[bold green]Indexed![/] {len(cat.tracks)} BGM tracks & {len(cat.sfx)} SFX clips -> {lib_dir / 'catalog.json'}"
    )


@music_app.command("list")
def music_list(
    library_dir: Optional[Path] = typer.Option(None, "--library-dir", "-d", help="Music library root directory"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Display indexed BGM tracks, moods, genres, BPM, energy, and LUFS loudness."""
    from src.audio.music_library import load_music_catalog

    cfg = load_config(config_path)
    lib_dir = library_dir or cfg.render.music_library_dir
    cat = load_music_catalog(lib_dir)
    if not cat or not cat.tracks:
        console.print(f"[bold yellow]No tracks indexed in {lib_dir}.[/] Run `uv run manga-recap music fetch` first.")
        return
    tbl = Table(title=f"Music Library ({len(cat.tracks)} BGM | {len(cat.sfx)} SFX)")
    tbl.add_column("Title")
    tbl.add_column("Moods")
    tbl.add_column("Genres")
    tbl.add_column("BPM", justify="right")
    tbl.add_column("Energy", justify="right")
    tbl.add_column("LUFS", justify="right")
    tbl.add_column("Dur(s)", justify="right")
    for tr in cat.tracks:
        tbl.add_row(
            tr.title[:34],
            ", ".join(tr.moods),
            ", ".join(tr.genres[:2]),
            f"{tr.bpm:.0f}",
            str(tr.energy),
            f"{tr.measured_lufs:.1f}",
            f"{tr.duration_sec:.0f}",
        )
    console.print(tbl)


@series_app.command("run")
def series_run(
    input_dir: Path = typer.Argument(..., help="Directory containing volume CBZs/ZIPs or volume subfolders"),
    series_id: str = typer.Option(..., "--series-id", "-s", help="Series identifier stored under series/<id>/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
    persona: Optional[str] = typer.Option(None, "--persona", "-p", help="Narration persona locked across the series"),
    tts_backend: Optional[str] = typer.Option(None, "--tts", help="Override TTS backend (edge_tts, vieneu)"),
    mode: str = typer.Option("episodic", "--mode", "-m", help="Continuity mode: episodic (separate episodes) | seamless (merge-ready)"),
    from_vol: int = typer.Option(1, "--from-vol", help="Start volume index (1-based)"),
    to_vol: Optional[int] = typer.Option(None, "--to-vol", help="End volume index (1-based, inclusive)"),
    compile_every: Optional[int] = typer.Option(None, "--compile-every", help="Auto-compile Arc video every N volumes"),
    compile_all: bool = typer.Option(False, "--compile-all", help="Auto-compile all completed volumes into 1 full-series MP4"),
    force: bool = typer.Option(False, "--force", "-f", help="Ignore cached volume checkpoints"),
    skip_render: bool = typer.Option(False, "--skip-render", help="Run stages 1-5 + exporters + Series Memory without MP4 render"),
) -> None:
    """Run an entire multi-volume manga series sequentially with Portrait Vault & 3-tier Series Memory."""
    from src.series.orchestrator import SeriesOrchestrator

    cfg = load_config(config_path)
    if tts_backend:
        cfg.tts.backend = tts_backend

    orch = SeriesOrchestrator(cfg)
    console.print(
        f"[bold cyan]Starting series run:[/] {series_id} (mode={mode}, from={from_vol}, to={to_vol or 'end'})"
    )
    rep = orch.run_series(
        input_dir=input_dir,
        series_id=series_id,
        continuity_mode=mode,
        persona=persona,
        from_vol=from_vol,
        to_vol=to_vol,
        compile_every=compile_every,
        compile_all=compile_all,
        force=force,
        skip_render=skip_render,
        on_volume_start=lambda v, tot, p, jid: console.print(
            f"[bold magenta]>>> Volume {v}/{tot}:[/] {p.name} -> job={jid}"
        ),
    )
    console.print(
        f"[bold green]Series complete![/] Processed {len(rep.volumes_processed)} volume(s): {rep.volumes_processed}"
    )
    console.print(f"[cyan]Glossary (editable anytime):[/] {Path(rep.series_dir) / 'glossary.yaml'}")
    for comp in rep.compilations:
        console.print(
            f"[bold green]Compiled video:[/] {comp.output_mp4} (vols={comp.volumes_included}, {comp.total_duration_sec:.1f}s)"
        )


@series_app.command("compile")
def series_compile(
    series_id: str = typer.Argument(..., help="Series identifier under series/<id>/"),
    volumes: Optional[str] = typer.Option(None, "--volumes", "-v", help="Volume range e.g. '1-6' or '1,2,3' (default: all)"),
    output_name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom output filename stem"),
    reencode: bool = typer.Option(False, "--reencode", help="Re-encode via libx264 instead of lossless stream copy"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Merge already-rendered volumes into 1 seamless Arc or Full-Series MP4 + SRT + YouTube Chapters + Credits."""
    from src.series.compiler import compile_series_videos, parse_volume_range
    from src.series.memory import SeriesStore

    cfg = load_config(config_path)
    store = SeriesStore(cfg.project.series_dir / series_id, series_id)
    if not store.state_path.exists():
        console.print(f"[bold red]Series not found:[/] {store.series_dir}")
        raise typer.Exit(code=1)

    state, _ = store.load_or_init()
    avail = [e.volume_index for e in state.ledger]
    selected = parse_volume_range(volumes, avail)
    if not selected:
        console.print(f"[bold yellow]No matching completed volumes.[/] Available: {avail}")
        raise typer.Exit(code=1)

    res = compile_series_videos(
        state=state,
        jobs_dir=cfg.project.jobs_dir,
        out_dir=store.compilations_dir,
        volumes=selected,
        output_name=output_name,
        reencode=reencode,
    )
    console.print(f"[bold green]Compiled MP4:[/] {res.output_mp4} ({res.total_duration_sec:.1f}s)")
    console.print(f"[cyan]Merged SRT:[/] {res.output_srt}")
    console.print(f"[cyan]YouTube Chapters:[/] {res.chapters_path}")
    console.print(f"[cyan]Music Credits:[/] {res.credits_path}")


@series_app.command("status")
def series_status(
    series_id: str = typer.Argument(..., help="Series identifier under series/<id>/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Inspect processed volumes, Portrait Vault characters, active plot threads, and glossary."""
    from src.series.memory import SeriesStore

    cfg = load_config(config_path)
    store = SeriesStore(cfg.project.series_dir / series_id, series_id)
    if not store.state_path.exists():
        console.print(f"[bold red]Series not found:[/] {store.series_dir}")
        raise typer.Exit(code=1)

    state, glossary = store.load_or_init()
    console.print(
        f"[bold cyan]Series:[/] {state.series_title or state.series_id} | "
        f"Volumes: {len(state.ledger)} | Characters: {len(state.characters)} | "
        f"Glossary terms: {len(glossary.terminology)} | Pronouns: {len(glossary.pronoun_pairs)}"
    )
    if state.ledger:
        t_vol = Table(title="Completed Volumes")
        t_vol.add_column("Vol", justify="right")
        t_vol.add_column("Job ID")
        t_vol.add_column("Title")
        t_vol.add_column("Dur (s)", justify="right")
        t_vol.add_column("MP4")
        for e in state.ledger:
            t_vol.add_row(
                str(e.volume_index),
                e.job_id,
                e.title_vi[:44],
                f"{e.duration_sec:.1f}",
                "OK" if (e.output_mp4 and Path(e.output_mp4).exists()) else "-",
            )
        console.print(t_vol)

    if state.characters:
        t_ch = Table(title="Series Cast & Portrait Vault")
        t_ch.add_column("ID")
        t_ch.add_column("Canonical Name")
        t_ch.add_column("Status")
        t_ch.add_column("Vols")
        t_ch.add_column("Revealed")
        t_ch.add_column("Portrait")
        for ch in state.characters[:24]:
            has_p = bool(ch.portrait_path and Path(ch.portrait_path).exists())
            t_ch.add_row(
                ch.char_id,
                ch.canonical_name,
                ch.status,
                f"{ch.first_seen_vol}..{ch.last_seen_vol}",
                f"v{ch.first_revealed_vol}" if ch.first_revealed_vol else "-",
                "YES" if has_p else "no",
            )
        console.print(t_ch)


@series_app.command("glossary")
def series_glossary(
    series_id: str = typer.Argument(..., help="Series identifier under series/<id>/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Display path and contents of the human-editable series glossary.yaml."""
    from src.series.memory import SeriesStore

    cfg = load_config(config_path)
    store = SeriesStore(cfg.project.series_dir / series_id, series_id)
    store.load_or_init()
    console.print(f"[bold cyan]Glossary path:[/] {store.glossary_path.resolve()}")
    console.print(store.glossary_path.read_text(encoding="utf-8"))


@series_app.command("import-job")
def series_import_job(
    series_id: str = typer.Argument(..., help="Target series identifier under series/<id>/"),
    job_dir: Path = typer.Argument(..., help="Path to an existing completed job directory (e.g. jobs/my_vol1)"),
    volume: int = typer.Option(1, "--vol", "-v", help="Volume index (1-based) for the imported job"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Bootstrap Series Memory & Portrait Vault from an already-completed volume job without re-running it."""
    from src.llm.client import LLMClient
    from src.series.memory import SeriesStore

    cfg = load_config(config_path)
    store = SeriesStore(cfg.project.series_dir / series_id, series_id)
    llm = LLMClient(cfg.llm)
    state, glossary = store.import_completed_job(job_dir=job_dir, volume_index=volume, llm=llm)
    portraits_saved = sum(1 for ch in state.characters if ch.portrait_path and Path(ch.portrait_path).exists())
    console.print(
        f"[bold green]Imported Vol {volume} from {job_dir} into series '{series_id}'![/]\n"
        f"  Characters: {len(state.characters)} ({portraits_saved} portraits in Vault)\n"
        f"  Active threads: {len(state.active_plot_threads)} | Terminology: {len(glossary.terminology)} | Pronouns: {len(glossary.pronoun_pairs)}\n"
        f"  Glossary: {store.glossary_path.resolve()}"
    )


if __name__ == "__main__":
    app()
