# Project Summary Report: Manga Recap Generator

## 1. Market Research & Optimal Architecture

### 1.1. Existing Market Solutions
- **Commercial SaaS / Desktop:** *RecapManga*, *Magarecap*, *ScrollBard*, *WTN Suite*, *TofuToon*, *ToonRecap Desktop* (VN). Strengths include fast webtoon strip slicing and CapCut timeline export; weaknesses include limited support for 2D right-to-left Manga reading order and "robotic narration syndrome" across multi-chapter arcs.
- **Open-Source:** *AniFlow*, *Manhwa-Recap*, *Manga2Animate*, *Magi v2*. Strong in CV/prototyping cores, but lack a multi-agent architecture with a cross-chapter Story Bible and a Critic verification gate.

### 1.2. Selected Architecture: Hybrid CV + Multi-Agent + Deterministic Render
Rather than delegating 100% of the pipeline to a VLM (which leads to misaligned panel crops, high API costs, and erratic video pacing), the system cleanly separates responsibilities:
1. **Traditional CV (ONNX YOLO Manga109 + OpenCV Gutter Slicer):** Pixel-accurate panel extraction, `face`/`body`/`text` detection, RTL/LTR/vertical reading-order sorting, and **Set-of-Mark (SoM)** image annotation (`[P1]`, `[P2]`, ...).
2. **Multi-Agent (`gpt-6-luna` + `gemini-3.8-flash-low-vision`):**
   - `PageAnalyzerAgent`: Reads SoM pages to extract actions, dialogue, and emotion per panel.
   - `StoryBibleAgent`: Maintains character profiles and plot threads across chapters.
   - `ScriptwriterAgent`: Writes narration scripts matching the chosen persona (`kich_tinh`, `hai_huoc`, `phan_tich`).
   - `CriticAgent`: Enforces valid `panel_id` grounding and script pacing invariants.
   - `ChoreographerAgent`: Assigns camera motions (`focus_face`, `zoom_in`, `zoom_out`, `shake_impact`) and subtitle cues.
3. **Deterministic Audio & Rendering:** VieNeu TTS (local Vietnamese) / Edge-TTS + FFmpeg/OpenCV rendering at `1280x720@24fps` with **AAC 48kHz stereo** audio, exporting SRT, CapCut Draft (`capcut_draft/draft_content.json`), and HTML Storyboard (`preview.html`).

---

## 2. Modular Structure (`uv` Environment)

- `pyproject.toml`: Dependency management & `manga-recap` CLI entrypoint via `uv`.
- `config.yaml`: Central configuration with `llm.default_model: gpt-6-luna`, `llm.vision_model: gemini-3.8-flash-low-vision`, `tts.backend: vieneu`, `render.audio_channels: 2`.
- `configs/personas.yaml`: Scriptwriter persona definitions.
- `src/core/`: `protocols.py`, `registry.py`, `telemetry.py`.
- `src/schemas/`: `ingestion.py`, `analysis.py`, `script.py`, `timeline.py`.
- `src/llm/client.py`: OpenAI-compatible client supporting text + vision, retries, telemetry, and `offline_fallback`.
- `src/cv/`: `loader.py` (image folders & CBZ/ZIP), `manga_detector.py`, `manhwa_slicer.py`, `som.py`, `stage.py`.
- `src/downloaders/nyaa.py`: Torrent download integration from `https://nyaa.si/view/<id>` via `libtorrent`, with volume filtering and automatic `.cbz` extraction.
- `src/agents/`: `page_analyzer.py`, `page_index.py`, `scriptwriter.py`, `critic.py`, `choreographer.py`.
- `src/audio/`: `base.py`, `vieneu_tts.py`, `edge_tts_backend.py`, `stage.py`.
- `src/render/` & `src/exporters/`: `compositor.py`, `subtitles.py`, `ffmpeg_renderer.py`, `capcut_draft.py`, `srt_exporter.py`, `html_preview.py`.

---

## 3. Implementation & Test Results

### 3.1. Automated Test Suite (`uv run pytest -q`)
- **13/13 baseline test cases PASSED** in `1.47s` (`tests/test_config_and_registry.py`, `tests/test_schemas_and_json.py`, `tests/test_cv.py`, `tests/test_audio.py`, `tests/test_renderer_and_exporters.py`, `tests/test_downloader.py`).

### 3.2. Audio Fix for `jobs/e2e_manga_multichap/06_output.mp4`
- Re-muxed audio from `04_voiceover.wav` to **AAC 48kHz stereo (`187 kb/s`)** and updated the `audio_channels: 2` default in `FFmpegOpenCVRenderer` (`mean_volume: -22.2 dB`, `max_volume: -4.1 dB`).

### 3.3. End-to-End Run: Pandora Hearts Vol 1 (`https://nyaa.si/view/1246645`)
- Downloaded and extracted from Nyaa.si into `downloads/pandora_hearts/extracted/`.
- Executed the full pipeline on **Pandora Hearts Vol 1** (`--job-id pandora_hearts_vol1 --max-pages 30 --tts vieneu`):
  - Stage 1 CV: Detected `120` panels across the first `30` pages.
  - Stage 4 TTS (`vieneu`, voice `Thiện Minh`): `101.36s`.
  - Stage 5–6 Render: `jobs/pandora_hearts_vol1/06_output.mp4` (`00:01:41.36`, `1280x720@24fps`, `aac 48kHz stereo`, `mean_volume: -22.0 dB`, `9,987,870 bytes`), alongside `06_subtitles.srt`, `capcut_draft/`, `preview.html`, and `telemetry.json`.
  - *Note:* When `OPENAI_API_KEY` / `VISION_API_KEY` are not configured in `.env`, Stages 2–3 execute via `offline_fallback`. Add API keys to `.env` to enable live vision analysis and scriptwriting via `gpt-6-luna` and `gemini-3.8-flash-low-vision`.
