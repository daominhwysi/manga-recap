"""End-to-end 6-stage Manga Recap Pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

from src.agents.choreographer import ChoreographerAgent
from src.agents.critic import CriticAgent
from src.agents.page_analyzer import PageAnalyzerAgent
from src.agents.page_index import PageIndexBuilder
from src.agents.scriptwriter import ScriptwriterAgent
from src.audio.stage import run_stage4_audio
from src.config import AppConfig
from src.core.telemetry import TelemetryTracker
from src.cv.stage import run_stage1_ingest
from src.exporters import EXPORTER_REGISTRY
from src.exporters import capcut_draft, html_preview, srt_exporter  # noqa: F401
from src.llm.client import LLMClient
from src.render import RENDERER_REGISTRY
from src.render import ffmpeg_renderer  # noqa: F401
from src.render.subtitles import write_srt


class MangaRecapPipeline:
    """Runs Stages 1-6 with checkpointing, PageIndex, telemetry, and modular backends."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(
        self,
        input_path: Path,
        job_id: str,
        persona: str | None = None,
        force: bool = False,
        skip_render: bool = False,
        prev_job_dir: Path | None = None,
    ) -> Path:
        job_dir = self.config.project.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        chosen_persona = persona or self.config.project.default_persona

        telemetry = TelemetryTracker(
            job_id=job_id,
            default_model=self.config.llm.default_model,
            vision_model=self.config.llm.vision_model,
        )
        llm = LLMClient(self.config.llm, telemetry=telemetry)

        # Stage 1: CV Ingest
        with telemetry.stage("stage1_cv_ingest") as d:
            manifest = run_stage1_ingest(
                job_id=job_id,
                input_path=input_path,
                job_dir=job_dir,
                cv_config=self.config.cv,
                force=force,
            )
            total_panels = sum(len(pg.panels) for ch in manifest.chapters for pg in ch.pages)
            d["chapters"] = len(manifest.chapters)
            d["panels"] = total_panels

        # Stage 2: Vision & PageIndex + StoryBible
        with telemetry.stage("stage2_vision_and_story_bible") as d:
            page_analyzer = PageAnalyzerAgent(llm)
            page_analyses = page_analyzer.run_all(
                manifest,
                job_dir,
                force=force,
                max_workers=self.config.script.vision_max_workers,
                batch_size=self.config.script.vision_batch_size,
                prev_job_dir=prev_job_dir,
            )
            index_builder = PageIndexBuilder(
                llm, pages_per_scene=self.config.script.pages_per_scene
            )
            tree, bible = index_builder.build(
                manifest,
                page_analyses,
                job_dir,
                force=force,
                prev_job_dir=prev_job_dir,
            )
            d["pages"] = len(page_analyses)
            d["scenes"] = tree.total_scenes
            d["tree_tiers"] = tree.tiers
            d["characters"] = len(bible.characters)

        # Stage 3: Scriptwriter & Closed-Loop Critic
        with telemetry.stage("stage3_script_and_critic") as d:
            writer = ScriptwriterAgent(llm, script_config=self.config.script)
            draft = writer.write_script(
                bible=bible,
                page_analyses=page_analyses,
                persona=chosen_persona,
                job_dir=job_dir,
                tree=tree,
                force=force,
            )
            critic = CriticAgent(llm, polish_seams=self.config.script.polish_seams)
            final_script, report = critic.review(
                draft,
                manifest,
                bible,
                job_dir,
                force=force,
                script_config=self.config.script,
                page_analyses=page_analyses,
            )
            d["beats"] = len(final_script.beats)
            d["critic_issues"] = len(report.issues)
            d["critic_score"] = report.score

        # Stage 4: TTS Audio
        with telemetry.stage("stage4_tts_audio") as d:
            audio_manifest = run_stage4_audio(
                final_script,
                job_dir=job_dir,
                tts_config=self.config.tts,
                force=force,
            )
            d["backend"] = audio_manifest.backend
            d["voice"] = audio_manifest.voice
            d["audio_duration_sec"] = round(audio_manifest.total_duration_ms / 1000.0, 2)

        # Stage 5: Choreographer
        with telemetry.stage("stage5_choreographer") as d:
            choreographer = ChoreographerAgent(self.config.render)
            timeline = choreographer.choreograph(
                job_id=job_id,
                manifest=manifest,
                script=final_script,
                audio_manifest=audio_manifest,
                job_dir=job_dir,
                force=force,
            )
            d["shots"] = len(timeline.shots)
            d["subtitles"] = len(timeline.subtitles)
            d["resolution"] = f"{timeline.width}x{timeline.height}@{timeline.fps}"

        srt_path = write_srt(timeline.subtitles, job_dir / "06_subtitles.srt")
        output_mp4 = job_dir / "06_output.mp4"

        # Stage 6a: Render MP4
        if not skip_render:
            with telemetry.stage("stage6a_render_mp4") as d:
                renderer = RENDERER_REGISTRY.create(
                    self.config.render.backend,
                    crf=self.config.render.crf,
                    preset=self.config.render.preset,
                    audio_bitrate=self.config.render.audio_bitrate,
                    audio_channels=self.config.render.audio_channels,
                    burn_subtitles=self.config.render.burn_subtitles,
                    margin_ratio=self.config.render.margin_ratio,
                    bottom_safe_ratio=self.config.render.bottom_safe_ratio,
                    bgm_volume=self.config.render.bgm_volume,
                    bgm_ducking=self.config.render.bgm_ducking,
                    subtitle_font_ratio=self.config.render.subtitle_font_ratio,
                    music_library_dir=self.config.render.music_library_dir,
                    min_bgm_cue_sec=self.config.render.min_bgm_cue_sec,
                )
                renderer.render(timeline, output_mp4, subtitles_path=srt_path)
                d["renderer"] = self.config.render.backend
                d["output_bytes"] = output_mp4.stat().st_size

        # Stage 6b: Exporters
        with telemetry.stage("stage6b_exporters") as d:
            exported: list[str] = []
            for exp_name in self.config.exporters.enabled:
                exporter = EXPORTER_REGISTRY.create(exp_name)
                exporter.export(timeline, job_dir)
                exported.append(exp_name)
            d["exporters"] = exported

        telemetry.save(job_dir / "telemetry.json")
        return output_mp4 if not skip_render else job_dir
