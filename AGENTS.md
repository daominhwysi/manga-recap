# AGENTS.md — Manga Recap Generator Architecture & Engineering Rules

> Đặc tả kiến trúc, filesystem, luồng dữ liệu 6-Stage và quy tắc bất biến cho mọi AI Agent phát triển `manga_recap_generator`. (Giữ dung lượng `< 12 KB` để hệ thống nạp trọn 100% vào `<user_rules>` mỗi lượt mà không bị cắt đuôi).

---

## 1. Mô Tả Dự Án & 3 Mục Tiêu Bất Biến

**Manga Recap Generator** là hệ thống pipeline tự động đầu-cuối chuyển đổi tập truyện tranh (**Manga** đọc phải-sang-trái hoặc **Manhwa/Webtoon** cuộn dọc, dạng thư mục ảnh hoặc `.cbz`/`.zip`) thành **video tóm tắt/review hoàn chỉnh (`.mp4` H.264 1280x720@24fps + Stereo AAC 48kHz)** bằng giọng đọc tiếng Việt tự nhiên, kèm phụ đề `.srt`, project CapCut và HTML preview.

1. **Đồng bộ Hình - Tiếng tuyệt đối (Strict Visual-Audio Grounding)**: Mỗi beat chỉ thuyết minh đúng hành động/hội thoại nằm trong 2–4 `panel_ids` gắn cho chính beat đó (phạm vi 1–3 trang liền kề, tiến đơn điệu theo chiều đọc, cấm kể vượt trước hình và cấm mượn panel sang trang chưa kể).
2. **Không lặp nội dung & Không rò rỉ phụ lục (Zero Duplication & Clean Filtering)**: Phân hoạch kịch bản tách biệt tuyệt đối `1 Scene = 1 Part`, tự động loại bỏ trang bìa, trang trắng, mục lục, trang đơn cắt trùng từ trang đôi (Double-Page Spread) và ngoại truyện/lời bạt cuối tập.
3. **Tổng quát hóa 100% (Zero Series-Specific Overfitting)**: Mọi thuật toán CV, xử lý ngôn ngữ và prompt đều dựa trên bất biến cấu trúc và đặc trưng hình ảnh/văn bản tổng quát, tuyệt đối không chứa tên nhân vật, câu thoại hay chỉ số trang cứng của bất kỳ bộ truyện cụ thể nào.

---

## 2. Quy Tắc Kỹ Thuật Bắt Buộc (Mandatory Engineering Rules)

### Rule 1 — Đồng bộ `AGENTS.md` khi sửa Workflow hoặc Filesystem (`AGENTS.md` Sync Rule)
- Bất kỳ khi nào thêm, xóa, đổi tên hoặc di chuyển file/thư mục trong `src/`, `configs/`, `tests/`, hoặc thay đổi kiến trúc Stage, schema dữ liệu hay file checkpoint trong `jobs/<job_id>/`, **BẮT BUỘC cập nhật Mục 4 (Filesystem) và Mục 5 (Workflow) của `AGENTS.md` ngay trong cùng tác vụ**.

### Rule 2 — Cấm tuyệt đối Anti-Pattern Overfit & Hardcode thiếu tính tổng quát
- **Cấm dữ liệu đặc thù bộ truyện**: Không đưa tên nhân vật, địa danh, tổ chức hay câu thoại mẫu của bất kỳ bộ truyện cụ thể nào (như *Pandora Hearts*) vào `src/`, `configs/` hay `config.yaml`.
- **Cấm ngưỡng chỉ số trang/bước nhảy cứng (Magic Cutoffs)**: Không dùng `idx < 6`, `idx in (0, 2, 3, 4)`, `idx >= int(total * 0.85)`, `range(7, ..., 8)` để đoán trang bìa, trang trắng, phụ lục hay ranh giới hồi; không tạo biến đếm cho một từ đơn lẻ (như `blood_count`).
- **Phương pháp chuẩn**: Bắt buộc dùng **máy trạng thái cấu trúc** (`while not story`), **đặc trưng điểm ảnh/hình học tổng quát** (`np.std < 4.5` cho trang trắng, MAE trên lưới `96x144` cho trang đôi cắt trùng, tỷ lệ nền trắng `> 75%` cho trang inset) và **thuật toán ngôn ngữ tổng quát** (so khớp 2/3-gram, scoped alias resolution). Mọi thay đổi phải pass `test_no_series_specific_overfitting_in_codebase`.

### Rule 3 — Kiểm thử & Git Commit theo từng tính năng (Atomic Feature Commit Rule)
- Sau khi hoàn thành bất kỳ tính năng, cải tiến hoặc sửa lỗi nào:
  1. Chạy `uv run pytest -q` (bắt buộc 100% PASS; thêm unit test mới cho logic mới).
  2. Cập nhật `AGENTS.md` nếu có thay đổi file/workflow (Rule 1).
  3. Commit ngay vào Git repository theo chuẩn Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).

### Rule 4 — Không nạp `RULES.md` vào Context
- File `RULES.md` (`110 KB`) ở thư mục gốc là tài liệu viết báo học thuật ML tiếng Anh (LaTeX, p-value, abstract) ngoài ngữ cảnh dự án này. **Tuyệt đối không đọc/nạp `RULES.md` vào context** để tránh tốn ~28.000 tokens. Các nguyên lý hữu ích (chỉ dẫn phủ định kèm ví dụ XẤU/TỐT, cấm cliché AI, giữ nhất quán `canonical_name`) đã tích hợp vào `configs/personas.yaml` và `src/agents/critic.py`.

---

## 3. Lệnh Vận Hành, Kiểm Thử & Quản Lý Cache

```bash
# Chạy toàn bộ unit tests (bao gồm quét chống overfitting codebase)
uv run pytest -v

# Chạy pipeline đầy đủ hoặc xem thông tin job
uv run manga-recap run <cbz_or_folder> --job-id <job_id> --persona kich_tinh
uv run manga-recap inspect jobs/<job_id>

# Xóa cache từ Stage 2b trở đi để tái tạo Index + Script + Audio + Video (giữ cache Vision 01 & 02a)
rm -rf jobs/<job_id>/02b_story_bible.json jobs/<job_id>/repo/tree_index.json \
       jobs/<job_id>/repo/MEMORY.md jobs/<job_id>/repo/notes \
       jobs/<job_id>/repo/OUTLINE.json jobs/<job_id>/repo/script \
       jobs/<job_id>/03* jobs/<job_id>/04* jobs/<job_id>/05* jobs/<job_id>/06*
```

---

## 4. Cấu Trúc Filesystem & Đặc Tả Chi Tiết Từng File

```text
manga_recap_generator/
├── AGENTS.md                        # Đặc tả kiến trúc, filesystem, workflow và quy tắc bắt buộc cho AI Agents
├── config.yaml                      # Cấu hình trung tâm (LLM provider/models, CV, Script, TTS, Render, Exporters)
├── pyproject.toml / uv.lock         # Dependencies Python (uv) và cấu hình pytest/ruff
├── .env.example / .gitignore        # Mẫu biến môi trường API và quy tắc loại trừ dữ liệu job lớn
├── configs/personas.yaml            # System Prompt và thông số giọng kể (kich_tinh, hai_huoc, phan_tich)
├── assets/bgm/default_dark_fantasy.wav # Nhạc nền D-minor stereo 48kHz tự tổng hợp (loop 64s không bản quyền)
├── models/manga109_yolo/            # Model ONNX YOLOv11 nhận diện frame, face, body, text
├── src/
│   ├── cli.py                       # Typer CLI (`run`, `download`, `inspect`)
│   ├── config.py                    # Pydantic schemas AppConfig; nạp config.yaml + ghi đè từ .env
│   ├── pipeline.py                  # Bộ điều phối 6 Stage (MangaRecapPipeline) kèm checkpointing & telemetry
│   ├── core/
│   │   ├── protocols.py             # Typing Protocol cho Detector, TTSBackend, Renderer, Exporter
│   │   ├── registry.py              # PluginRegistry quản lý đăng ký các backend theo tên
│   │   └── telemetry.py             # Theo dõi thời gian chạy từng stage, lượng token, fallback và cảnh báo
│   ├── schemas/
│   │   ├── ingestion.py             # Schemas Stage 1: BBox, FocalPoint, PanelInfo, PageInfo, ChapterInfo, JobManifest
│   │   ├── analysis.py              # Schemas Stage 2: PanelAnalysis, PageAnalysis, CharacterProfile, ChapterSummary, StoryBible
│   │   ├── script.py                # Schemas Stage 3: ScriptBeat, RecapScript, CriticIssue, CriticReport
│   │   └── timeline.py              # Schemas Stage 4-5: AudioBeatInfo, AudioManifest, TimelineShot, SubtitleCue, VideoTimeline
│   ├── llm/client.py                # LLMClient chuẩn OpenAI API: tách provider text/vision (`extra_body`), circuit breaker, Pydantic JSON
│   ├── cv/
│   │   ├── registry.py / loader.py  # DETECTOR_REGISTRY; giải nén .cbz/.zip hoặc đọc thư mục theo `natural_sort_key`
│   │   ├── manga_detector.py        # MangaYoloDetector (ONNX): phát hiện frame/face/body/text, lọc mảnh rác, sắp xếp RTL/LTR
│   │   ├── manhwa_slicer.py         # ManhwaStripSlicer: cắt dải ảnh dọc webtoon theo khoảng trắng ngang (`row_std < 6.0`)
│   │   ├── som.py / stage.py        # Vẽ nhãn Set-of-Mark ([P1]..[Pn]) và điều phối Stage 1 (`run_stage1_ingest`, tự chọn theo `h/w >= 2.2`)
│   ├── agents/
│   │   ├── page_analyzer.py         # Stage 1.5 & 2a: Pre-Pass tìm nhân vật + ghép `cast_sheet.jpg`; lọc trang trắng (`np.std < 4.5`); Chunked Parallel Rolling Vision
│   │   ├── page_index.py            # Stage 2b: Chuẩn hóa bí danh (`scoped_map` chống spoiler); lọc trang đôi cắt trùng (`96x144` MAE); lọc front/backmatter theo trạng thái; dựng `PageIndexTree`, `StoryBible`, `MEMORY.md`
│   │   ├── story_bible.py           # Tiện ích truy vấn và hợp nhất hồ sơ nhân vật StoryBible
│   │   ├── scriptwriter.py          # Stage 3a: Biên kịch 3 tầng với phân hoạch `1 Scene = 1 Part` tách biệt, bảng phân bổ trang theo beat và khóa thứ tự panel đơn điệu (`<= 3` trang/beat)
│   │   ├── critic.py                # Stage 3b: Kiểm định kịch bản (grounding, biên giới Part, khoảng trống >10 trang, nhảy lùi trang, ôm >4 trang, lặp nội dung N-gram, từ sáo rỗng AI)
│   │   └── choreographer.py         # Stage 5: Đạo diễn hình ảnh — ghép diptych cùng trang, mượn panel cùng trang (`beat_pages`), chia thời lượng shot theo độ dài câu đọc, Ken Burns và chia phụ đề
│   ├── audio/
│   │   ├── registry.py / base.py    # TTS_REGISTRY, hash cache theo từng beat, đọc thời lượng WAV và nối `04_voiceover.wav`
│   │   ├── edge_tts_backend.py      # Microsoft Edge TTS tiếng Việt (`vi-VN-NamMinhNeural`) xuất WAV PCM 48kHz
│   │   ├── vieneu_tts.py            # Backend VieNeu TTS nội bộ (tùy chọn)
│   │   ├── bgm_generator.py / stage.py # Tự tổng hợp nhạc nền Dark Fantasy D-minor stereo 48kHz (`ensure_default_bgm`) & điều phối Stage 4
│   ├── render/
│   │   ├── registry.py / compositor.py # RENDERER_REGISTRY; kết xuất frame OpenCV (nền mờ bokeh 1/4, bóng đổ, diptych RTL, pan/zoom/shake)
│   │   ├── subtitles.py / preview_html.py # Xuất `.srt`, vẽ hộp phụ đề UTF-8 có cache RGBA pre-multiplied, xuất preview HTML
│   │   └── ffmpeg_renderer.py       # Kết xuất MP4 đa luồng qua pipe FFmpeg (VAAPI/libx264), tự cắt viền trắng thừa (`_trim_excess_white_border`) và trộn BGM sidechain ducking
│   ├── exporters/                   # `registry.py`, `srt_exporter.py`, `capcut_draft.py` (`draft_content.json`), `html_preview.py`
│   └── downloaders/nyaa.py          # Tìm kiếm và tải torrent/magnet manga từ Nyaa
└── tests/                           # 7 file pytest: `test_schemas_and_json.py`, `test_config_and_registry.py`, `test_cv.py`, `test_page_index_and_agents.py`, `test_audio.py`, `test_renderer_and_exporters.py`, `test_downloader.py`
```

### Cấu trúc Checkpoint trong `jobs/<job_id>/`
- **Stage 1**: `extracted/` (ảnh trang gốc), `panels/*.jpg` (ảnh cắt từng khung), `som/*_som.jpg` (ảnh trang đóng dấu `[P1]..[Pn]`), `01_manifest.json` (`JobManifest`).
- **Stage 1.5 & 2a**: `repo/seed_cast.json`, `repo/cast_sheet.jpg` (bảng ghép mặt nhân vật chuẩn), `02a_pages/*.json` (cache Vision từng trang), `02a_all_pages.json`.
- **Stage 2b**: `repo/notes/scene_*.md` (ghi chú Markdown từng scene kèm `ar` và `face+text`), `repo/tree_index.json` (`PageIndexTree` + `skipped_pages`), `02b_story_bible.json` (`StoryBible`), `repo/MEMORY.md`.
- **Stage 3a & 3b**: `repo/OUTLINE.json` (`1 Scene = 1 Part`), `repo/script/parts/part_*.json`, `repo/script/part_boundaries.json`, `03a_script_draft.json`, `03b_script_final.json`, `03b_critic_report.json`.
- **Stage 4, 5 & 6**: `audio/beats/*.wav`, `04_voiceover.wav`, `04_audio_manifest.json`, `05_timeline.json` (`VideoTimeline`), `06_subtitles.srt`, `06_output.mp4`, `exports/`, `telemetry.json`.

---

## 5. Kiến Trúc & Luồng Xử Lý Chi Tiết 6-Stage Pipeline

1. **Stage 1 — CV Ingest (`src/cv/`)**:
   - Giải nén `.cbz`/`.zip` theo `natural_sort_key`. Tự động chọn `ManhwaStripSlicer` nếu `h / w >= 2.2`, ngược lại chọn `MangaYoloDetector` (ONNX YOLOv11 phát hiện `frame`, `face`, `body`, `text`).
   - Loại bỏ khung rác (`min_side < 95px`, dải screentone trống không có mặt/chữ hoặc `crop_std < 18.0`), sắp xếp khung theo hàng ngang (chồng lấp dọc `> 35%`) theo chiều đọc `rtl`/`ltr` và vẽ nhãn Set-of-Mark `[P1]..[Pn]`.
2. **Stage 1.5 & 2a — Pre-Pass Cast Sheet & Rolling Vision (`src/agents/page_analyzer.py`)**:
   - **Chặn trang trắng bằng CV**: `_is_blank_or_solid_page()` kiểm tra `np.std(img) < 4.5`; nếu trắng/đen trơn thì trả về ngay `[PHỤ LỤC] Trang trắng` mà không gọi Vision LLM để chống bịa nội dung.
   - **Pre-Pass**: Chọn 16 trang giàu mặt người và chữ nhất trải đều qua 6 phân vùng để trích xuất nhân vật có câu thoại bằng chứng (`evidence_quote`) và dựng ảnh `repo/cast_sheet.jpg`.
   - **Chunked Parallel Rolling Vision**: Đọc ảnh theo batch 4 trang song song; mỗi batch luôn đính kèm `cast_sheet.jpg` ở ảnh đầu tiên cùng `RollingSceneState` (nhân vật đang có mặt + tóm tắt 3 trang trước).
3. **Stage 2b — PageIndex Tree & StoryBible (`src/agents/page_index.py`)**:
   - **Scoped Alias Resolution**: Hợp nhất tên nhân vật về `canonical_name`. Mọi nhãn mô tả vô danh (bắt đầu bằng `[` hoặc chứa `"chưa rõ"`, `"bí ẩn"`, `"người lạ"`) chỉ được ánh xạ cục bộ trong `scoped_map` theo `page_ids`, cấm đưa vào `global_map` để chống lộ spoiler danh tính ở đầu truyện.
   - **Khử trang đơn cắt trùng từ trang đôi (`_detect_duplicate_spread_pages`)**: Với mỗi trang đôi ngang (`w > h * 1.15`) có 2 trang dọc kế tiếp `p1, p2`, cắt đôi trang ngang, resize về `96x144` grayscale và so sánh MAE với `p1, p2` theo cả chiều RTL và LTR; nếu `min(rtl_diff, ltr_diff) < 12.0` thì đưa `p1, p2` vào `skipped_pages`.
   - **Lọc Frontmatter/Backmatter theo trạng thái (`_filter_story_pages`)**: Khi `not story` (chưa gặp trang truyện thật đầu tiên), lọc toàn bộ bìa/mục lục/trang 1-panel không thoại qua `_is_frontmatter_page()`. Mọi trang có tiền tố `[phụ lục]` hoặc khớp `_BACKMATTER_KEYWORDS` sau khi vào truyện đều bị loại khỏi `SceneNode`.
4. **Stage 3a — Disjoint Multi-Part Scriptwriter (`src/agents/scriptwriter.py`)**:
   - **Phân hoạch `1 Scene = 1 Part` tách biệt**: Mỗi `SceneNode` (~20 trang) ánh xạ 1-1 thành một `OutlinePart` với `target_beats` và `target_words` tỷ lệ thuận theo số trang (~6–7 beats/part). Không chồng lấn trang giữa các Part và không cắt đuôi `handoff` gây kể lặp.
   - **Per-Beat Page Pacing Guide**: Chia đều danh sách trang của Part cho `target_beats` (~2–3 trang/beat) ngay trong prompt để LLM đi đều đến trang cuối Part, không dồn cục cuối hồi.
   - **Khóa thứ tự Panel đơn điệu (`_sanitize_part_beats`)**: Giữ `panel_id` trong `allowed_set`, ép con trỏ `cursor_rank` tiến đơn điệu và giới hạn mỗi beat tối đa `<= 3` trang liền kề.
5. **Stage 3b — Critic Verification (`src/agents/critic.py`)**:
   - Kiểm tra 6 chốt chặn: (1) Panel Grounding có fallback theo vị trí tương đối, (2) Ranh giới `allowed_panels` của từng Part, (3) Độ rộng 1 beat `> 4` trang, (4) Nhảy lùi trang `cur_min < prev_page_pos`, (5) Khoảng trống bỏ cách `> 10` trang, (6) Trùng lặp nội dung giữa các beat qua tập 2/3-grams (`_content_ngrams`, bắt lỗi khi trùng `>= 4` cụm và `>= 35%`) cùng danh sách `_BANNED_CLICHES`.
6. **Stage 4, 5 & 6 — Audio, Choreographer & FFmpeg Renderer (`src/audio/`, `src/agents/choreographer.py`, `src/render/`)**:
   - **Audio**: Tổng hợp WAV PCM 48kHz theo mã băm từng beat và nối thành `04_voiceover.wav` + `04_audio_manifest.json`.
   - **Choreographer**: Chỉ ghép diptych khi 2 panel dọc thuộc **cùng một trang (`p1.page_id == p2.page_id`)**; khi beat dài cần thêm hình, chỉ mượn panel chưa dùng thuộc **cùng các trang của beat đó (`cand.page_id in beat_pages`)**; chia thời lượng từng shot tỷ lệ thuận với độ dài câu đọc (`slot_durations`).
   - **FFmpeg Renderer**: Tự cắt viền trắng thừa cho các trang inset có `> 75%` diện tích trắng (`_trim_excess_white_border`), dựng frame bokeh + phụ đề đa luồng và đẩy qua FFmpeg (`h264_vaapi` hoặc `libx264`) kèm trộn nhạc nền `sidechaincompress` ducking.
