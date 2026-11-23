"""Application configuration loader combining config.yaml and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "manga_recap_generator"
    jobs_dir: Path = Path("jobs")
    downloads_dir: Path = Path("downloads")
    default_language: str = "vi"
    default_persona: str = "kich_tinh"


class LLMConfig(BaseModel):
    provider: str | None = None
    vision_provider: str | None = None
    default_model: str = "gpt-6.2-sol"
    fallback_model: str | None = "gpt-6-luna"
    vision_model: str = "gemini-3.8-flash-low-vision"
    temperature: float = 0.4
    max_tokens: int = 32768
    timeout_sec: int = 120
    max_retries: int = 2
    offline_fallback: bool = True
    base_url: str | None = None
    api_key: str | None = None
    vision_base_url: str | None = None
    vision_api_key: str | None = None


class CVConfig(BaseModel):
    detector: str = "auto"
    yolo_model_path: Path = Path("models/manga109_yolo/v2023.12.07_m_yv11/model.onnx")
    yolo_labels_path: Path = Path("models/manga109_yolo/v2023.12.07_m_yv11/labels.json")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    reading_order: Literal["rtl", "ltr", "vertical"] = "rtl"
    max_pages_per_chapter: int | None = None


class TTSConfig(BaseModel):
    backend: str = "edge_tts"
    voice: str = "vi-VN-NamMinhNeural"
    vieneu_voice: str = "Thiện Minh"
    rate: str = "+5%"
    sample_rate: int = 48000


class RenderConfig(BaseModel):
    backend: str = "ffmpeg_opencv"
    aspect_ratio: str = "16:9"
    width: int = 1280
    height: int = 720
    fps: int = 24
    crf: int = 22
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    audio_channels: int = 2
    max_shot_duration_ms: int = 3500
    burn_subtitles: bool = True
    subtitle_font_ratio: float = 0.036
    zoom_speed_per_sec: float = 0.055
    min_zoom_delta: float = 0.06
    max_zoom_delta: float = 0.18
    face_zoom_bonus: float = 0.04
    margin_ratio: float = 0.06
    bottom_safe_ratio: float = 0.14
    enable_transitions: bool = True
    enable_diptych: bool = True
    bgm_path: str | None = None
    bgm_volume: float = 0.14
    bgm_ducking: bool = True
    music_library_dir: Path = Path("assets/music_library")
    min_bgm_cue_sec: float = 28.0


class ExportersConfig(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["srt", "capcut_draft", "html_preview"])


class DownloaderConfig(BaseModel):
    nyaa_base_url: str = "https://nyaa.si"
    timeout_sec: int = 600
    seed_after_download: bool = False


class ScriptConfig(BaseModel):
    model_config = {"extra": "ignore"}

    target_duration_min: float = 9.0
    pages_per_scene: int = 20
    fast_path_max_pages: int = 30
    vision_batch_size: int = 4
    vision_max_workers: int = 4
    words_per_minute: int = 225
    polish_seams: bool = True
    decoupled_two_pass: bool = True
    story_temperature: float = 0.72
    align_temperature: float = 0.15


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    script: ScriptConfig = Field(default_factory=ScriptConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    exporters: ExportersConfig = Field(default_factory=ExportersConfig)
    downloader: DownloaderConfig = Field(default_factory=DownloaderConfig)


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ and val:
            os.environ[key] = val


def load_config(config_path: Path | str = "config.yaml") -> AppConfig:
    """Load AppConfig from YAML file and overlay environment variables."""
    _load_dotenv(Path(".env"))
    path = Path(config_path)
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded

    cfg = AppConfig.model_validate(raw)

    # Environment overrides
    if env_prov := os.environ.get("LLM_PROVIDER"):
        cfg.llm.provider = env_prov
    if env_vprov := os.environ.get("VISION_LLM_PROVIDER"):
        cfg.llm.vision_provider = env_vprov
    if env_model := os.environ.get("DEFAULT_LLM_MODEL"):
        cfg.llm.default_model = env_model
    if env_vision := os.environ.get("VISION_LLM_MODEL"):
        cfg.llm.vision_model = env_vision
    if env_key := os.environ.get("OPENAI_API_KEY"):
        cfg.llm.api_key = env_key
    if env_base := os.environ.get("OPENAI_BASE_URL"):
        cfg.llm.base_url = env_base
    if env_vkey := os.environ.get("VISION_API_KEY"):
        cfg.llm.vision_api_key = env_vkey
    if env_vbase := os.environ.get("VISION_BASE_URL"):
        cfg.llm.vision_base_url = env_vbase

    return cfg


def load_personas(path: Path | str = "configs/personas.yaml") -> dict[str, dict[str, Any]]:
    """Load persona definitions from YAML."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return dict(data.get("personas", {}))
