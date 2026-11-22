# Báo Cáo Tổng Kết Dự Án: Manga Recap Generator

## 1. Nghiên cứu thị trường & Phương án tối ưu

### 1.1. Các giải pháp hiện có trên thị trường
- **SaaS / Desktop thương mại:** *RecapManga*, *Magarecap*, *ScrollBard*, *WTN Suite*, *TofuToon*, *ToonRecap Desktop* (VN). Điểm mạnh là cắt dải webtoon nhanh, xuất timeline CapCut; điểm yếu là xử lý Manga 2D đọc phải→trái còn hạn chế và dễ bị "hội chứng robot" khi giữ mạch nhiều chương.
- **Open-Source:** *AniFlow*, *Manhwa-Recap*, *Manga2Animate*, *Magi v2*. Mạnh ở lõi CV/prototyping nhưng thiếu kiến trúc multi-agent có Story Bible xuyên chương và Critic kiểm duyệt.

### 1.2. Phương án kiến trúc tối ưu đã chọn: Hybrid CV + Multi-Agent + Deterministic Render
Thay vì giao 100% cho VLM (dễ cắt khung lệch, tốn chi phí API, sai nhịp video), hệ thống phân vai rõ ràng:
1. **CV Truyền thống (ONNX YOLO Manga109 + OpenCV Gutter Slicer):** Tách khung chính xác từng pixel, phát hiện `face`/`body`, sắp xếp thứ tự đọc RTL/LTR/vertical và sinh ảnh **Set-of-Mark (SoM)** đánh số `[P1]`, `[P2]`...
2. **Multi-Agent (`gpt-6-luna` + `gemini-3.8-flash-low-vision`):**
   - `PageAnalyzerAgent`: Đọc ảnh SoM hiểu hành động, hội thoại, cảm xúc từng panel.
   - `StoryBibleAgent`: Duy trì hồ sơ nhân vật & mạch cốt truyện xuyên nhiều chương.
   - `ScriptwriterAgent`: Viết kịch bản theo persona (`kich_tinh`, `hai_huoc`, `phan_tich`).
   - `CriticAgent`: Kiểm tra grounding `panel_id` hợp lệ và nhịp độ câu chữ.
   - `ChoreographerAgent`: Phân bổ shot camera (`focus_face`, `zoom_in`, `zoom_out`, `shake_impact`) và phụ đề.
3. **Audio & Render xác định:** VieNeu TTS (local tiếng Việt) / Edge-TTS + FFmpeg/OpenCV render `1280x720@24fps` kèm audio **AAC 48kHz stereo**, đồng thời xuất SRT, CapCut Draft (`capcut_draft/draft_content.json`) và HTML Storyboard (`preview.html`).

---

## 2. Cấu trúc Module hóa (Môi trường `uv`)

- `pyproject.toml`: Quản lý dependencies & CLI entrypoint `manga-recap` bằng `uv`.
- `config.yaml`: Cấu hình trung tâm với `llm.default_model: gpt-6-luna`, `llm.vision_model: gemini-3.8-flash-low-vision`, `tts.backend: vieneu`, `render.audio_channels: 2`.
- `configs/personas.yaml`: Khai báo các persona biên kịch.
- `src/core/`: `protocols.py`, `registry.py`, `telemetry.py`.
- `src/schemas/`: `ingestion.py`, `analysis.py`, `script.py`, `timeline.py`.
- `src/llm/client.py`: Client chuẩn OpenAI-compatible hỗ trợ text + vision, retry, telemetry và `offline_fallback`.
- `src/cv/`: `loader.py` (hỗ trợ folder ảnh & CBZ/ZIP), `manga_detector.py`, `manhwa_slicer.py`, `som.py`, `stage.py`.
- `src/downloaders/nyaa.py`: Tích hợp tải torrent từ `https://nyaa.si/view/<id>` qua `libtorrent`, lọc theo volume và tự động giải nén `.cbz`.
- `src/agents/`: `page_analyzer.py`, `story_bible.py`, `scriptwriter.py`, `critic.py`, `choreographer.py`.
- `src/audio/`: `base.py`, `vieneu_tts.py`, `edge_tts_backend.py`, `stage.py`.
- `src/render/` & `src/exporters/`: `compositor.py`, `subtitles.py`, `ffmpeg_renderer.py`, `capcut_draft.py`, `srt_exporter.py`, `html_preview.py`.

---

## 3. Kết quả triển khai & Kiểm thử

### 3.1. Bộ kiểm thử tự động (`uv run pytest -q`)
- **13/13 test cases PASSED** trong `1.47s` (`tests/test_config_and_registry.py`, `tests/test_schemas_and_json.py`, `tests/test_cv.py`, `tests/test_audio.py`, `tests/test_renderer_and_exporters.py`, `tests/test_downloader.py`).

### 3.2. Sửa lỗi âm thanh `jobs/e2e_manga_multichap/06_output.mp4`
- Đã re-mux âm thanh từ `04_voiceover.wav` sang chuẩn **AAC 48kHz stereo (`187 kb/s`)** và cập nhật mặc định `audio_channels: 2` trong `FFmpegOpenCVRenderer` (`mean_volume: -22.2 dB`, `max_volume: -4.1 dB`).

### 3.3. Tải & tạo video review Pandora Hearts Vol 1 (`https://nyaa.si/view/1246645`)
- Đã tải và giải nén thành công từ Nyaa.si vào `downloads/pandora_hearts/extracted/`.
- Đã chạy pipeline hoàn chỉnh cho **Pandora Hearts Vol 1** (`--job-id pandora_hearts_vol1 --max-pages 30 --tts vieneu`):
  - Stage 1 CV: Phát hiện `120` panels từ `30` trang đầu.
  - Stage 4 TTS (`vieneu`, giọng `Thiện Minh`): `101.36s`.
  - Stage 5–6 Render: `jobs/pandora_hearts_vol1/06_output.mp4` (`00:01:41.36`, `1280x720@24fps`, `aac 48kHz stereo`, `mean_volume: -22.0 dB`, `9,987,870 bytes`), kèm `06_subtitles.srt`, `capcut_draft/`, `preview.html`, `telemetry.json`.
  - *Lưu ý:* Do môi trường hiện tại chưa cấu hình `OPENAI_API_KEY` / `VISION_API_KEY` trong `.env`, Stage 2–3 chạy qua `offline_fallback`. Chỉ cần thêm API key vào `.env` để kích hoạt phân tích hình ảnh và viết kịch bản trực tiếp qua `gpt-6-luna` và `gemini-3.8-flash-low-vision`.
