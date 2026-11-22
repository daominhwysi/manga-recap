"""Typer CLI entrypoint for manga-recap."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from src.config import load_config
from src.downloaders.nyaa import NyaaDownloader
from src.pipeline import MangaRecapPipeline

app = typer.Typer(help="Automated multi-agent Manga & Manhwa Recap Video Generator")
console = Console()


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Path to manga folder, multi-chapter folder, or CBZ file"),
    job_id: str = typer.Option("demo_job", "--job-id", "-j", help="Output job identifier under jobs/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
    persona: Optional[str] = typer.Option(None, "--persona", "-p", help="Narration persona"),
    tts_backend: Optional[str] = typer.Option(None, "--tts", help="Override TTS backend (edge_tts, vieneu)"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="Limit pages per chapter"),
    force: bool = typer.Option(False, "--force", "-f", help="Ignore cached checkpoints"),
    skip_render: bool = typer.Option(False, "--skip-render", help="Run stages 1-5 + exporters without MP4 render"),
) -> None:
    """Run the full 6-stage manga recap pipeline."""
    cfg = load_config(config_path)
    if tts_backend:
        cfg.tts.backend = tts_backend
    if max_pages is not None:
        cfg.cv.max_pages_per_chapter = max_pages

    console.print(f"[bold cyan]Starting job:[/] {job_id} (model={cfg.llm.default_model})")
    pipeline = MangaRecapPipeline(cfg)
    result = pipeline.run(
        input_path=input_path,
        job_id=job_id,
        persona=persona,
        force=force,
        skip_render=skip_render,
    )
    console.print(f"[bold green]Done![/] Output: {result}")


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


if __name__ == "__main__":
    app()
