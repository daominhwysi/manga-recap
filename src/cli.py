"""Typer CLI entrypoint for manga-recap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.agents.product_evaluator import ProductEvaluatorAgent
from src.config import load_config
from src.downloaders.nyaa import NyaaDownloader
from src.llm.client import LLMClient
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
    prev_job: Optional[Path] = typer.Option(None, "--prev-job", help="Optional previous volume job dir for cast continuity"),
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
        prev_job_dir=prev_job,
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
def evaluate(
    job_dir: Path = typer.Argument(..., help="Path to a job directory under jobs/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Run deterministic Script + Visual QA only (0 tokens)"),
) -> None:
    """Evaluate final script + visual assets as a unified product and write 07_EVAL_REPORT.md."""
    if not job_dir.exists() or not (job_dir / "03b_script_final.json").exists():
        console.print(f"[bold red]Missing job or 03b_script_final.json in:[/] {job_dir}")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)
    llm = None if no_llm else LLMClient(cfg.llm)
    evaluator = ProductEvaluatorAgent(llm=llm, script_config=cfg.script)
    report = evaluator.evaluate(job_dir, use_llm=not no_llm)

    status_color = "green" if report.passed else "yellow"
    console.print(
        f"[bold {status_color}]Product Eval ({report.job_id}):[/] "
        f"score={report.overall_score}/100 | passed={report.passed} | "
        f"findings={len(report.findings)}"
    )
    console.print(
        f"  Script: {report.script_metrics.total_beats} beats, {report.script_metrics.total_words}/"
        f"{report.script_metrics.target_words} words, quote_grounding={report.script_metrics.quote_grounding_rate}%"
    )
    console.print(
        f"  Visual: coverage={report.visual_metrics.story_page_coverage_pct}%, "
        f"shots={report.visual_metrics.total_timeline_shots}, freeze={len(report.visual_metrics.visual_freeze_shots)}"
    )
    console.print(f"  Report: {job_dir / '07_EVAL_REPORT.md'}")
    if report.contact_sheet_path:
        console.print(f"  Contact sheet: {report.contact_sheet_path}")


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
