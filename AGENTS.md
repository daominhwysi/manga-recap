# AGENTS.md — Manga Recap Generator Architecture & Engineering Rules

> Đặc tả kiến trúc, filesystem, luồng dữ liệu 6-Stage và quy tắc bất biến cho mọi AI Agent phát triển `manga_recap_generator`.

---

## 1. Mô Tả Dự Án & 3 Mục Tiêu Bất Biến

**Manga Recap Generator** là hệ thống pipeline tự động đầu-cuối chuyển đổi tập truyện tranh (**Manga** đọc phải-sang-trái hoặc **Manhwa/Webtoon** cuộn dọc, dạng thư mục ảnh hoặc `.cbz`/`.zip`) thành **video tóm tắt/review hoàn chỉnh (`.mp4` H.264 + Stereo AAC)** bằng giọng đọc tiếng Việt tự nhiên, kèm phụ đề `.srt`, project CapCut và HTML preview.

1. **Đồng bộ Hình - Tiếng tuyệt đối (Strict Visual-Audio Grounding)**: Mỗi beat chỉ thuyết minh đúng hành động/hội thoại nằm trong các `panel_ids` gắn cho chính beat đó (tiến đơn điệu theo chiều đọc, độ rộng trang mỗi beat tự co giãn theo mật độ `pages / beats`, cấm kể vượt trước hình và cấm mượn panel sang trang chưa kể).
2. **Không lặp nội dung & Không rò rỉ phụ lục (Zero Duplication & Clean Filtering)**: Phân hoạch kịch bản tách biệt tuyệt đối `1 Scene = 1 Part`, tự động loại bỏ trang bìa, trang trắng, mục lục, trang đơn cắt trùng từ trang đôi (Double-Page Spread) và ngoại truyện/lời bạt cuối tập.
3. **Tổng quát hóa 100% (Zero Series-Specific Overfitting)**: Mọi thuật toán CV, xử lý ngôn ngữ và prompt đều dựa trên bất biến cấu trúc và tỷ lệ hình học/văn bản tương đối, tuyệt đối không chứa tên nhân vật, câu thoại, kích thước pixel tuyệt đối hay chỉ số trang cứng của bất kỳ bộ truyện cụ thể nào.

---

## 2. Quy Tắc Kỹ Thuật Bắt Buộc (Mandatory Engineering Rules)

### Rule 1 — Kiểm tra `docs/ARCHITECTURE.md` & Đồng bộ `AGENTS.md` khi sửa Kiến trúc, Workflow hoặc Filesystem
- Trước và sau khi sửa đổi kiến trúc, thuật toán hoặc cấu trúc dữ liệu của bất kỳ Stage nào, **BẮT BUỘC mở `docs/ARCHITECTURE.md` (Mục 6 — Bảng Kiểm Tra Tác Động Kiến Trúc) để kiểm tra các bất biến liên đới và xem cần cập nhật gì không**.
- Bất kỳ khi nào thêm, xóa, đổi tên hoặc di chuyển file/thư mục trong `src/`, `configs/`, `docs/`, `tests/`, hoặc thay đổi kiến trúc Stage, schema dữ liệu hay file checkpoint trong `jobs/<job_id>/`, **BẮT BUỘC cập nhật đồng bộ cả `docs/ARCHITECTURE.md` và Mục 4 (Filesystem) + Mục 5 (Workflow) của `AGENTS.md` ngay trong cùng tác vụ**.

### Rule 2 — Cấm tuyệt đối Anti-Pattern Overfit & Hardcode thiếu tính tổng quát
- **Cấm dữ liệu đặc thù bộ truyện**: Không đưa tên nhân vật, địa danh, tổ chức hay câu thoại mẫu của bất kỳ bộ truyện cụ thể nào vào `src/`, `configs/` hay `config.yaml`.
- **Cấm ngưỡng chỉ số trang/pixel tuyệt đối cứng (Magic Cutoffs)**: Không dùng chỉ số trang cứng (`idx < 6`, `idx in (...)`, tỷ lệ vị trí trang cố định), kích thước pixel tuyệt đối không theo tỷ lệ trang, bước nhảy beat cố định hay biến đếm cho một từ đơn lẻ.
- **Phương pháp chuẩn**: Bắt buộc dùng **máy trạng thái cấu trúc** (`while not story`), **tỷ lệ hình học/thống kê ảnh tương đối theo trang** (kích thước panel tối thiểu tỷ lệ theo cạnh ngắn của trang `min(w, h)`, độ lệch chuẩn mức xám cho trang trắng, sai số điểm ảnh MAE trên ảnh thu nhỏ đồng tỷ lệ cho trang đôi cắt trùng, tỷ lệ diện tích nền trắng cho trang inset) và **mật độ kịch bản động** (số trang tối đa mỗi beat và khoảng nhảy trang tính theo `pages / target_beats`, so khớp N-gram, scoped alias resolution). Mọi thay đổi phải pass `test_no_series_specific_overfitting_in_codebase`.

### Rule 3 — Kiểm thử & Git Commit theo từng tính năng (Atomic Feature Commit Rule)
- Sau khi hoàn thành bất kỳ tính năng, cải tiến hoặc sửa lỗi nào:
  1. Chạy `uv run pytest -q` (bắt buộc 100% PASS; thêm unit test mới cho logic mới).
  2. Cập nhật `AGENTS.md` nếu có thay đổi file/workflow (Rule 1).
  3. Commit ngay vào Git repository theo chuẩn Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).

### Rule 4 — Không nạp `RULES.md` vào Context
- File `RULES.md` ở thư mục gốc là tài liệu viết báo học thuật ML tiếng Anh (LaTeX, p-value, abstract) ngoài ngữ cảnh dự án này. **Tuyệt đối không đọc/nạp `RULES.md` vào context**. Các nguyên lý hữu ích (chỉ dẫn phủ định kèm ví dụ XẤU/TỐT, cấm cliché AI, giữ nhất quán `canonical_name`) đã tích hợp vào `configs/personas.yaml` và `src/agents/critic.py`.

### Rule 5 — Khóa Sự Thật & Kỹ Thuật, Thả Lỏng Nghệ Thuật Kể (Invariants vs. Affordances)
- **Chỉ áp luật cứng cho Sự thật & Kỹ thuật**: grounding hình-tiếng, không bịa/spoiler sớm, không lặp nội dung, sạch phụ lục, chuẩn âm thanh/phụ đề.
- **Cung cấp Bộ công cụ thay vì Công thức ép buộc (Affordances over Mandates)**: Với cấu trúc kể chuyện (kiểu Hook mở đầu, ranh giới Scene, độ dài câu/beat, chèn bình luận/hài, thời lượng đứng hình nghệ thuật), **cấm biến phong cách tham khảo thành khuôn cứng bắt buộc cho mọi tác phẩm**. Pipeline phải cho phép Agent chọn chiến lược phù hợp với chính bộ truyện đang xử lý.
- **Kiểm định ở tầm Phân phối thay vì Vi mô**: Critic bắt lỗi sự thật và triệu chứng toàn cục (ví dụ: cả tập đều tăm tắp một nhịp, bỏ sót/dồn trang), không ép từng beat/shot phải khớp một khuôn mẫu số học cố định.

---

## 3. Lệnh Vận Hành, Kiểm Thử & Quản Lý Cache

```bash
# Chạy toàn bộ unit tests (bao gồm quét chống overfitting codebase)
uv run pytest -v

# Chạy pipeline đầy đủ (đơn tập hoặc theo chuỗi Series nhiều tập) và xem thông tin job/series
uv run manga-recap run <cbz_or_folder> --job-id <job_id> --persona kich_tinh
uv run manga-recap inspect jobs/<job_id>
uv run manga-recap series import-job <series_id> jobs/<vol1_job> --vol 1
uv run manga-recap series run <folder_24_vols> --series-id <series_id> --mode episodic --compile-every 4 --compile-all
uv run manga-recap series compile <series_id> --volumes 1-6
uv run manga-recap series status <series_id>
uv run manga-recap series glossary <series_id>

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
├── AGENTS.md                        # Đặc tả kiến trúc tóm tắt, filesystem, workflow và quy tắc bắt buộc cho AI Agents
├── docs/
│   ├── ARCHITECTURE.md              # Tài liệu kiến trúc chi tiết tiếng Việt (mục lục, giải thích 6 Stage, bất biến & bảng kiểm tra tác động)
│   ├── ARCHITECTURE_REVIEW.md       # Báo cáo kiểm toán toàn diện kiến trúc, 15 lỗ hổng, đối chiếu kịch bản YouTube & lộ trình V2
│   └── FINAL_REPORT.md              # Báo cáo tổng kết thực nghiệm pipeline
├── config.yaml                      # Cấu hình trung tâm (LLM provider/models, CV, Script, TTS, Render, Exporters)
├── pyproject.toml / uv.lock         # Dependencies Python (uv) và cấu hình pytest/ruff
├── .env.example / .gitignore        # Mẫu biến môi trường API và quy tắc loại trừ dữ liệu job lớn
├── configs/personas.yaml            # System Prompt và thông số giọng kể (kich_tinh, hai_huoc, phan_tich)
├── assets/bgm/default_dark_fantasy.wav # Nhạc nền D-minor stereo tự tổng hợp (loop không bản quyền)
├── models/manga109_yolo/            # Model ONNX YOLOv11 nhận diện frame, face, body, text
├── src/
│   ├── cli.py                       # Typer CLI (`run`, `download`, `inspect`, `music fetch|index|list`, `series run|compile|status|glossary`)
│   ├── config.py                    # Pydantic schemas AppConfig (+`project.series_dir`); nạp config.yaml + ghi đè từ .env (`default_model=gpt-6.2-sol`)
│   ├── pipeline.py                  # Bộ điều phối 6 Stage (MangaRecapPipeline) kèm checkpointing, `--prev-job`, `series_ctx` (Series Memory) & telemetry
│   ├── core/
│   │   ├── protocols.py             # Typing Protocol cho Detector, TTSBackend, Renderer, Exporter
│   │   ├── registry.py              # PluginRegistry quản lý đăng ký các backend theo tên
│   │   └── telemetry.py             # Theo dõi thời gian chạy từng stage, lượng token, fallback và cảnh báo
│   ├── schemas/
│   │   ├── ingestion.py             # Schemas Stage 1: BBox, FocalPoint, PanelInfo, PageInfo, ChapterInfo, JobManifest
│   │   ├── analysis.py              # Schemas Stage 2: PanelAnalysis, PageAnalysis (+`visual_verify_code`, `verbatim_quote_vi`, `visual_punch`), CharacterProfile (+`first_named_page_id`), StoryBible
│   │   ├── script.py                # Schemas Stage 3: `VALID_MOODS`, `normalize_mood`, ScriptBeat (+`beat_type`, `tts_text`), RecapScript (+`hook_strategy`), CriticIssue, CriticReport
│   │   ├── series.py                # Schemas Multi-Volume Series Continuity: `SeriesCharacter`, `VolumeHandoff`, `VolumeLedgerEntry`, `SeriesGlossary`, `SeriesState`, `SeriesContext`
│   │   └── timeline.py              # Schemas Stage 4-5: WordTimestamp, AudioBeatInfo, AudioManifest, TimelineShot, SubtitleCue, VideoTimeline (+`reading_order`)
│   ├── series/
│   │   ├── memory.py                # `SeriesStore`: Persistent Portrait Vault (`portraits/`), chọn cast động theo tập (`select_active_cast_for_volume`), Bộ nhớ 3 tầng O(1) (`Hot/Warm/Cold`), `glossary.yaml` sửa tay & `update_after_volume`
│   │   ├── compiler.py              # `compile_series_videos`: ghép MP4 nhiều tập theo Arc/Full-Series + dịch mốc thời gian `.srt` + xuất YouTube Chapters + gộp `music_credits.txt`
│   │   └── orchestrator.py          # `SeriesOrchestrator` & `discover_series_volumes`: tự quét Vol 1..N theo `natural_sort_key`, chạy nối tiếp có checkpoint từng tập, tự nạp `glossary.yaml` giữa các tập & tùy chọn `--compile-every`/`--compile-all`
│   ├── llm/client.py                # LLMClient chuẩn OpenAI API: `complete_json` & `complete_xml` (+`temperature_override`), data-URL ảnh, circuit breaker (ngưỡng 2 lần lỗi)
│   ├── cv/
│   │   ├── registry.py / loader.py  # DETECTOR_REGISTRY; giải nén .cbz/.zip hoặc đọc thư mục theo `natural_sort_key`
│   │   ├── manga_detector.py        # MangaYoloDetector (ONNX): Recursive XY-Cut RTL/LTR (chuẩn L-shape), cắt viền trắng inset đồng bộ `focal_points` ngay Stage 1
│   │   ├── manhwa_slicer.py         # ManhwaStripSlicer: cắt dải ảnh dọc webtoon theo ngưỡng tỷ lệ chiều rộng + hậu kiểm YOLO `face/body/text`
│   │   ├── som.py / stage.py        # Vẽ nhãn Set-of-Mark ([P1]..[Pn]) và điều phối Stage 1 (`run_stage1_ingest`)
│   ├── agents/
│   │   ├── alignment.py             # Semantic & Quote-Pinned Panel Alignment dùng chung Stage 3a & 3b: Monotonic DP pre-anchor map, Quote Pinning, monotonicity-aware `verify_beats_alignment` (`quote_mismatch`, `visual_desync`) & `repair_part_panel_alignment_deterministic`
│   │   ├── page_analyzer.py         # Stage 1.5 & 2a (XML): Pre-Pass cast sheet UTF-8; đóng dấu Visual Verify Code (RAM) chống rơi ảnh/lệch trang; khử spread trùng CV; Adaptive Batching; `verbatim_quote_vi` & `visual_punch`
│   │   ├── page_index.py            # Stage 2b (Summarizer XML): Chuẩn hóa bí danh (`scoped_map`); tính `first_named_page_id` chống spoil tên; lọc spread 2 chiều; Semantic Scene Segmentation; StoryBible (`pronunciation_map` mặc định trống); `scoped_outline_view` O(N)
│   │   ├── scriptwriter.py          # Stage 3a (Decoupled 2-Pass XML): Pass 1 (`ProseWriterAgent`, `temperature=0.72`) viết văn Video Essay liền mạch theo 3-4 Chương lớn từ bản tóm tắt sạch không rác CV/panel_id; Pass 2 (`PanelAlignerAgent`, `temperature=0.15`) chia beat giữ nguyên văn gốc kèm Bản đồ Neo Trang + Auto-Retry Gate chống trượt pha (`verify_beats_alignment`)
│   │   ├── critic.py                # Stage 3b (Critic V3.7): Deterministic Gate + Semantic Visual-Audio Alignment Gate + Surgical Panel Repair (`CriticPanelRepairer` + Rollback Guard + deterministic safety-net) + Opener/Global Discourse Tic Scanner + Read-Only Auditor + Allowlisted Text Repairer
│   │   └── choreographer.py         # Stage 5: Đạo diễn hình ảnh — word-timestamp sync, tách 2 nhịp sub-shot (`Wide -> Focus Face`), LTR/RTL & 9:16 vertical, đồng bộ 7 `mood`
│   ├── audio/
│   │   ├── registry.py / base.py    # TTS_REGISTRY, `apply_pronunciation_glossary`, `beat_cache_key` (gồm `rate`), anti-aliased Fourier resampling & word timestamp alignment
│   │   ├── edge_tts_backend.py      # Microsoft Edge TTS tiếng Việt kèm `WordBoundary` timestamps & tổng hợp song song
│   │   ├── vieneu_tts.py            # Backend VieNeu TTS nội bộ kèm fallback tự động sang Edge-TTS
│   │   ├── music_library.py         # Tự động hóa Dataset Âm Nhạc & SFX (`assets/music_library/catalog.json`): tải bộ starter CC-BY 4.0 (Incompetech), phân tích DSP (LUFS, BPM, brightness, intro-skip) & auto-tag custom folder
│   │   ├── music_director.py        # Smart Music Director: chọn bảng màu thể loại theo `StoryBible`, gom cue nhạc theo Scene (>=28s), chuẩn hóa -16 LUFS, equal-power crossfade 2.5s, SFX cooldown & xuất `music_credits.txt`
│   │   ├── bgm_generator.py / stage.py # Điều phối BGM (ưu tiên `music_director` nếu có catalog, fallback 4-bar synth) & điều phối Stage 4
│   ├── render/
│   │   ├── registry.py / compositor.py # RENDERER_REGISTRY; kết xuất frame OpenCV (bokeh, shadow, diptych RTL/LTR, Ken Burns/shake)
│   │   ├── subtitles.py / preview_html.py # Xuất `.srt`, phụ đề word-wrap UTF-8 (9:16/16:9) có `subtitle_font_ratio`, HTML preview kèm ảnh thumbnail & audio player
│   │   └── ffmpeg_renderer.py       # Kết xuất MP4 đa luồng (VAAPI/libx264), tự nạp dynamic multi-mood BGM + sidechaincompress ducking
│   ├── exporters/                   # `registry.py`, `srt_exporter.py`, `capcut_draft.py` (đủ diptych + BGM + subtitles), `html_preview.py`
│   └── downloaders/nyaa.py          # Tìm kiếm và tải torrent/magnet manga từ Nyaa
└── tests/                           # 8 file pytest: `test_schemas_and_json.py`, `test_config_and_registry.py`, `test_cv.py`, `test_page_index_and_agents.py`, `test_series_continuity.py`, `test_audio.py`, `test_renderer_and_exporters.py`, `test_downloader.py`
```

### Cấu trúc Checkpoint trong `jobs/<job_id>/` & Bộ nhớ Chuỗi Tập `series/<series_id>/`
- **Stage 1**: `extracted/` (ảnh trang gốc), `panels/*.jpg` (ảnh cắt từng khung), `som/*_som.jpg` (ảnh trang đóng dấu `[P1]..[Pn]`), `01_manifest.json` (`JobManifest`).
- **Stage 1.5 & 2a**: `repo/seed_cast.json` (+`portrait_path`), `repo/cast_sheet.jpg` (bảng ghép mặt nhân vật chuẩn động 16 ô), `02a_pages/*.json` (cache Vision từng trang), `02a_all_pages.json`.
- **Stage 2b**: `repo/notes/scene_*.md` (ghi chú Markdown từng scene kèm tỷ lệ `ar` và cờ `face+text`), `repo/tree_index.json` (`PageIndexTree` + `skipped_pages`), `02b_story_bible.json` (`StoryBible`), `repo/MEMORY.md`.
- **Stage 3a & 3b**: `repo/OUTLINE.json` (`1 Scene = 1 Part`), `repo/script/acts/act_*.json` & `repo/script/prose_draft.md` (Pass 1 thuần văn), `repo/script/parts/part_*.json` (Pass 2 đã gắn panel), `repo/script/part_boundaries.json`, `03a_script_draft.json`, `03b_script_final.json`, `03b_critic_report.json`.
- **Stage 4, 5 & 6**: `audio/beats/*.wav`, `04_voiceover.wav`, `04_audio_manifest.json`, `audio/series_music_context.json`, `audio/04b_bgm_dynamic.wav`, `audio/04b_music_cues.json`, `05_timeline.json` (`VideoTimeline`), `06_subtitles.srt`, `06_output.mp4`, `exports/` (+`music_credits.txt`), `telemetry.json`.
- **Series Continuity (`series/<series_id>/`)**: `series_state.json` (`SeriesState`: cast xuyên tập, `ledger` 24+ tập, `last_handoff`, `active_plot_threads`, `music_genre_prefs`), `glossary.yaml` (file YAML cho phép người dùng sửa tay thuật ngữ/xưng hô/tên nhân vật giữa các tập), `portraits/<char_id>.jpg` (Portrait Vault bền vững), `compilations/*.mp4|*.srt|*_chapters.txt|*_music_credits.txt` (video gộp Arc/Full-Series).

---

## 5. Kiến Trúc & Luồng Xử Lý Chi Tiết 6-Stage Pipeline

1. **Stage 1 — CV Ingest (`src/cv/`)**:
   - Giải nén `.cbz`/`.zip` theo `natural_sort_key`. Tự động chọn `ManhwaStripSlicer` (ngưỡng tương đối theo chiều rộng ảnh + hậu kiểm YOLO `face/body/text`) cho webtoon hoặc `MangaYoloDetector` (ONNX YOLOv11 + sắp xếp **Recursive XY-Cut** chuẩn bố cục L-shape RTL/LTR) cho trang sách.
   - Cắt viền trắng thừa trang inset và dịch chuyển tọa độ `focal_points` ngay tại Stage 1 (`trim_white_border_with_offset`), rồi vẽ nhãn Set-of-Mark `[P1]..[Pn]`.
2. **Stage 1.5 & 2a — Pre-Pass Cast Sheet & Rolling Vision (`src/agents/page_analyzer.py`)**:
   - **Khử trang trắng & trang đôi cắt trùng trước khi gọi Vision**: `_is_blank_or_solid_page()` + `_detect_duplicate_spread_pages()` (kiểm tra cả 2 chiều `[Spread,P1,P2]` và `[P1,P2,Spread]`) chặn bằng CV thuần (0 token) trước khi đưa vào Vision LLM.
   - **Pre-Pass UTF-8 & Cross-Volume**: Quét trang mẫu có bằng chứng thoại (`evidence_quote`), ghép bảng `repo/cast_sheet.jpg` bằng Pillow UTF-8, tùy chọn kế thừa `--prev-job`.
   - **Adaptive Panel Batching & Rich Visuals**: Gom batch thích ứng theo mật độ khung (`~16–18 panels/call`) kèm `cast_sheet.jpg` và `RollingSceneState`, trích thêm `verbatim_quote_vi` + `visual_punch`.
3. **Stage 2b — PageIndex Tree & StoryBible (`src/agents/page_index.py`)**:
   - **Scoped Alias Resolution Toàn Văn Bản**: Chuẩn hóa tên trong `characters_present`, `page_summary`, `action_summary` và `dialogue_summary`. Mọi nhãn vô danh giữ cục bộ trong `scoped_map` chống lộ spoiler.
   - **Semantic Scene Segmentation + Nén Notes**: Tách cảnh tự nhiên kèm `narrative_weight` (1–5) & `transition_type` (có kiểm tra phủ rời rạc 100% trang), nén cú pháp `scene_*.md` (bỏ giới hạn 24k), gom batch 3 scenes/call có cache từng scene và nạp `seed_cast.json` + sinh `pronunciation_map` (`repo/pronunciation.json`).
4. **Stage 3a — Decoupled 2-Pass Scriptwriter (`src/agents/scriptwriter.py`)**:
   - **Phân hoạch `1 Scene = 1 Part` & Water-Filling**: `OutlinePlannerAgent` lập dàn ý ngữ nghĩa (`skim`/`steady`/`spotlight`) hòa trộn qua `_allocate_part_budgets` (Water-filling + Largest Remainder).
   - **Pass 1 — Pure Storyteller (`ProseWriterAgent`, `temperature=0.72`)**: Gom các hồi nối tiếp thành **3–4 Chương lớn (`_group_parts_into_acts`)**, nạp bản tóm tắt sạch không rác CV/panel_id (`_strip_cv_metadata_for_prose`), viết văn **Video Essay tiếng Việt liền mạch, giàu nhạc tính** (ZERO `panel_id`/XML beat tax, bỏ bệnh ép rắc `"nha/kìa/anh em"`, vẫn khóa chống spoil tên theo `first_named_page_id`), lưu `repo/script/prose_draft.md`.
   - **Pass 2 — Panel Aligner (`PanelAlignerAgent`, `temperature=0.15`)**: Được cấp **Bản đồ Neo Trang (`build_alignment_anchors`, Monotonic DP + Quote Pin)** trước khi chạy; chỉ cắt nguyên văn từng câu của Pass 1 vào các `<beat>` đúng nhịp thở và gán `1–4 panel_id` KHỚP ĐÚNG SỰ KIỆN/THOẠI tiến đơn điệu, có **Auto-Retry Gate (`verify_beats_alignment`)** chống trượt pha (`quote_mismatch`/`visual_desync`), kiểm tra `_prose_preservation_ratio >= 0.78`, `pin_missing_quote_panels` và `_sanitize_part_beats`.
5. **Stage 3b — Critic V3.7: Deterministic Gate, Semantic Alignment Gate, Surgical Panel Repair, Global Tic Scanner, Read-Only Auditor & Surgical Text Repair (`src/agents/critic.py`)**:
   - **Tầng 1 (Python)**: Kiểm định hard invariants (`grounding`, biên giới Part, lùi trang, `coverage_gap`, **đồng bộ Hình–Tiếng nội dung monotonicity-aware `verify_beats_alignment`**, `spoiler_name`, N-gram `duplication`, cliché AI, `budget > +30%`) + quét lặp thống kê (`opener_repeat`, **`tic_repeat`** bắt khẩu ngữ thừa ở *mọi vị trí* trong câu trừ lời thoại trích dẫn, `seam_echo`, và cảnh báo vĩ mô mật độ dấu `:` / câu cụt staccato).
   - **Tầng 2 & 3a (Visual Surgical Panel Repair `CriticPanelRepairer`)**: Khi phát hiện `panel_sync` / `coverage_gap` còn sót từ Stage 3a, tự căn lại `panel_ids` trong allowlist từng Part (kèm **Rollback Guard** bắt buộc giảm lỗi và không lùi trang + lưới an toàn tất định `repair_part_panel_alignment_deterministic`) mà giữ nguyên 100% `narration_vi`.
   - **Tầng 2 & 3b (Read-Only `CriticSemanticAuditor` + Surgical Text Repair `CriticSeamPolisher`)**: Đọc liền mạch toàn tập trong 1 context để chỉ điểm tối đa 8 beat lặp ý khác chữ hoặc sượng điểm chuyển hồi; sửa văn theo **allowlist `editable_ids`** có **Per-Beat Rollback Guard**, sau đó điền `tts_text` theo `pronunciation_map`.
6. **Stage 4, 5 & 6 — Audio, Smart Music Director, Choreographer & FFmpeg Renderer (`src/audio/`, `src/agents/choreographer.py`, `src/render/`)**:
   - **Audio & Music Dataset**: `manga-recap music fetch/index/list` quản lý kho nhạc CC-BY/custom tại `assets/music_library/catalog.json` (tự đo DSP: LUFS, BPM, brightness, intro-skip). **`Smart Music Director` (`music_director.py`)** chọn bảng màu thể loại theo `StoryBible.tone_summary`, gom nhạc thành các **Music Cues dài >= 28s** tránh nhảy bài mỗi beat, chuẩn hóa `-16 LUFS`, trộn **2.5s equal-power crossfade** + SFX thật (cooldown 6s) + tự xuất `exports/music_credits.txt` (fallback mượt về 4-bar synth nếu chưa có thư viện).
   - **Choreographer & Renderer**: Word-timestamp subtitle sync, word-wrap phụ đề cho cả `16:9` & `9:16`, tách 2 nhịp sub-shot (`Wide -> Focus Face`) trên slot dài giàu chi tiết, hỗ trợ RTL/LTR ở `compositor.py`, HTML preview có `<img>` + `<audio>`, và CapCut draft đầy đủ diptych + BGM + subtitles.
7. **Multi-Volume Series Continuity Engine (`src/series/`, `src/schemas/series.py`)**:
   - **Portrait Vault & Dynamic Active Cast**: Lưu ảnh mặt nhân vật chuẩn (`series/<id>/portraits/<char_id>.jpg`), chọn động tối đa 16 nhân vật hoạt động của Vol $k$ (ưu tiên nhân vật cục bộ + `last_handoff` + hoạt động gần đây, loại nhân vật đã chết/rời sân lâu) và mở khóa `first_named_page_id` cho nhân vật đã lộ tên ở Vol $< k$.
   - **Bộ Nhớ 3 Tầng $O(1)$ (`Hot / Warm / Cold`) + Human-Editable `glossary.yaml`**: Nạp `last_handoff` (Vol $k-1$), `active_plot_threads` + `terminology`/`pronoun_pairs` cố định, và truy xuất có chọn lọc tối đa 3 mốc `ledger` tập cũ liên quan để gọi callback mà không phình token qua 24+ tập.
   - **2 Chế Độ Nối Tập (`episodic` vs `seamless`) & Series Compiler**: `episodic` tạo video riêng lẻ có cầu nối gọn 1–2 câu từ tập trước; `seamless` tắt lời chào mở/kết giữa các tập để ghép liền mạch qua `manga-recap series compile` thành 1 video Arc / Full-Series kèm `.srt` dịch mốc thời gian, YouTube Chapters và `music_credits.txt` hợp nhất. Khóa bảng màu nhạc `music_genre_prefs` và tránh lặp lại track mở đầu vừa dùng ở tập trước.
