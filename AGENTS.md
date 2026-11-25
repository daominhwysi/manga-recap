# AGENTS.md — Manga Recap Generator Architecture & Engineering Rules

> Concise specification of the architecture, filesystem layout, 6-stage data flow, and mandatory invariants for all AI Agents developing `manga_recap_generator`.

---

## 1. Project Overview & 3 Core Invariants

**Manga Recap Generator** is an end-to-end automated pipeline that converts comic volumes (right-to-left **Manga** or vertical-scrolling **Manhwa/Webtoon**, as image folders or `.cbz`/`.zip` archives) into **complete recap/review videos (`.mp4` H.264 + Stereo AAC)** with natural Vietnamese narration, `.srt` subtitles, CapCut project drafts, and HTML previews.

1. **Strict Visual-Audio Grounding**: Every beat may only narrate actions and dialogue present in the `panel_ids` attached to that beat (advancing monotonically in reading order, with per-beat page span scaling dynamically by `pages / beats`, forbidding future spoilers or borrowing panels from unnarrated pages).
2. **Zero Duplication & Clean Filtering**: Script partitioning strictly enforces `1 Scene = 1 Part` with disjoint page sets, automatically filtering covers, blank pages, tables of contents, single pages sliced redundantly from double-page spreads, and end-of-volume extras/afterwords.
3. **100% Generalization (Zero Series-Specific Overfitting)**: All CV algorithms, NLP heuristics, and prompts rely solely on structural invariants and relative geometric/text ratios—never containing character names, dialogue quotes, absolute pixel thresholds, or hardcoded page indices from any specific series.

---

## 2. Mandatory Engineering Rules

### Rule 1 — Check `docs/ARCHITECTURE.md` & Sync `AGENTS.md` on Architecture, Workflow, or Filesystem Changes
- Before and after modifying the architecture, algorithms, or data contracts of any Stage, **you MUST consult `docs/ARCHITECTURE.md` (Section 6 — Architecture Impact Checklist)** to verify downstream invariants and required updates.
- Whenever you add, delete, rename, or move files/directories in `src/`, `configs/`, `docs/`, or `tests/`, or change Stage architecture, data schemas, or `jobs/<job_id>/` checkpoints, **you MUST update both `docs/ARCHITECTURE.md` and Sections 4 (Filesystem) + 5 (Workflow) of `AGENTS.md` in the same task**.

### Rule 2 — Zero Overfitting & No Hardcoded Heuristics
- **No series-specific data**: Never put character names, locations, organizations, or sample dialogue from any specific series into `src/`, `configs/`, or `config.yaml`.
- **No magic page-index/pixel cutoffs**: Never use hardcoded page indices (`idx < 6`, `idx in (...)`, fixed page position ratios), unscaled pixel thresholds, fixed beat-step limits, or single-word counters.
- **Standard approach**: Always use **structural state machines** (`while not story`), **page-relative geometric/statistical ratios** (minimum panel size relative to `min(w, h)`, grayscale std-dev for blank pages, downscaled MAE for duplicate spreads, white background area ratio for inset pages), and **dynamic script density** (max pages per beat and page jump thresholds derived from `pages / target_beats`, N-gram matching, scoped alias resolution). All changes must pass `test_no_series_specific_overfitting_in_codebase`.

### Rule 3 — Testing & Atomic Feature Git Commits
- After completing any feature, improvement, or bugfix:
  1. Run `uv run pytest -q` (100% PASS required; add unit tests for new logic).
  2. Update `AGENTS.md` if files or workflows changed (Rule 1).
  3. Commit immediately to Git using Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).

### Rule 4 — Lock Truth & Technique, Unlock Storytelling Craft (Invariants vs. Affordances)
- **Apply hard rules only to Truth & Engineering**: visual-audio grounding, no fabrication/early spoilers, zero content duplication, clean appendix filtering, and audio/subtitle standards.
- **Provide Affordances over Mandates**: For narrative craft (opening hook strategy, scene boundaries, sentence/beat cadence, commentary/humor, artistic freeze durations), **never turn reference style habits into rigid mandates for every series**. The pipeline must allow Agents to choose strategies fitting the work being adapted.
- **Critique at the Distribution Level rather than Micro-Constraints**: The Critic catches factual errors and global symptoms (e.g., uniform pacing across the whole volume, skipped or bunched pages), rather than forcing individual beats/shots into a fixed arithmetic template.

---

## 3. Operations, Testing & Cache Management Commands

```bash
# Run full unit test suite (including codebase anti-overfitting scan)
uv run pytest -v

# Run full pipeline (single volume or multi-volume series) and inspect job/series status
uv run manga-recap run <cbz_or_folder> --job-id <job_id> --persona kich_tinh
uv run manga-recap inspect jobs/<job_id>
uv run manga-recap series import-job <series_id> jobs/<vol1_job> --vol 1
uv run manga-recap series run <folder_24_vols> --series-id <series_id> --mode episodic --compile-every 4 --compile-all
uv run manga-recap series compile <series_id> --volumes 1-6
uv run manga-recap series status <series_id>
uv run manga-recap series glossary <series_id>

# Clear caches from Stage 2b onward to regenerate Index + Script + Audio + Video (preserving Vision 01 & 02a)
rm -rf jobs/<job_id>/02b_story_bible.json jobs/<job_id>/repo/tree_index.json \
       jobs/<job_id>/repo/MEMORY.md jobs/<job_id>/repo/notes \
       jobs/<job_id>/repo/OUTLINE.json jobs/<job_id>/repo/script \
       jobs/<job_id>/03* jobs/<job_id>/04* jobs/<job_id>/05* jobs/<job_id>/06*
```

---

## 4. Filesystem Structure & File Specifications

```text
manga_recap_generator/
├── AGENTS.md                        # Concise architecture spec, filesystem map, workflow, and mandatory rules for AI Agents
├── docs/
│   ├── ARCHITECTURE.md              # Detailed architecture doc (TOC, 6-Stage deep dive, invariants & impact checklist)
│   ├── ARCHITECTURE_REVIEW.md       # Comprehensive architecture audit, 17 findings, YouTube benchmark & V2 roadmap
│   └── FINAL_REPORT.md              # Experimental pipeline summary report
├── config.yaml                      # Central config (LLM provider/models, CV, Script, TTS, Render, Exporters)
├── pyproject.toml / uv.lock         # Python dependencies (uv) and pytest/ruff config
├── .env.example / .gitignore        # API env template and exclusion rules for large job artifacts
├── configs/personas.yaml            # System prompts and narration settings (kich_tinh, hai_huoc, phan_tich)
├── assets/bgm/default_dark_fantasy.wav # Synthesized royalty-free D-minor stereo BGM loop
├── models/manga109_yolo/            # YOLOv11 ONNX model for frame, face, body, and text detection
├── src/
│   ├── cli.py                       # Typer CLI (`run`, `download`, `inspect`, `music fetch|index|list`, `series run|compile|status|glossary`)
│   ├── config.py                    # Pydantic AppConfig schemas (+`project.series_dir`); loads config.yaml + .env overrides (`default_model=gpt-6.2-sol`)
│   ├── pipeline.py                  # 6-Stage orchestrator (MangaRecapPipeline) with checkpointing, `--prev-job`, `series_ctx` & telemetry
│   ├── core/
│   │   ├── protocols.py             # Typing Protocols for Detector, TTSBackend, Renderer, Exporter
│   │   ├── registry.py              # PluginRegistry managing backend registration by name
│   │   └── telemetry.py             # Stage timing, token usage, fallback, and warning tracker
│   ├── schemas/
│   │   ├── ingestion.py             # Stage 1 schemas: BBox, FocalPoint, PanelInfo, PageInfo, ChapterInfo, JobManifest
│   │   ├── analysis.py              # Stage 2 schemas: PanelAnalysis, PageAnalysis (+`visual_verify_code`, `verbatim_quote_vi`, `visual_punch`), CharacterProfile (+`first_named_page_id`), StoryBible
│   │   ├── script.py                # Stage 3 schemas: `VALID_MOODS`, `normalize_mood`, ScriptBeat (+`beat_type`, `tts_text`), RecapScript (+`hook_strategy`), CriticIssue, CriticReport
│   │   ├── series.py                # Multi-Volume Series Continuity schemas: `SeriesCharacter`, `VolumeHandoff`, `VolumeLedgerEntry`, `SeriesGlossary`, `SeriesState`, `SeriesContext`
│   │   └── timeline.py              # Stage 4-5 schemas: WordTimestamp, AudioBeatInfo, AudioManifest, TimelineShot, SubtitleCue, VideoTimeline (+`reading_order`)
│   ├── series/
│   │   ├── memory.py                # `SeriesStore`: Persistent Portrait Vault (`portraits/`), dynamic active cast selection (`select_active_cast_for_volume`), O(1) 3-tier memory (`Hot/Warm/Cold`), human-editable `glossary.yaml` & `update_after_volume`
│   │   ├── compiler.py              # `compile_series_videos`: Arc/Full-Series MP4 compilation + cumulative `.srt` time shifting + YouTube Chapters + merged `music_credits.txt`
│   │   └── orchestrator.py          # `SeriesOrchestrator` & `discover_series_volumes`: natural-sorted Vol 1..N discovery, checkpointed sequential runs, live `glossary.yaml` reloading & `--compile-every`/`--compile-all`
│   ├── llm/client.py                # OpenAI-compatible LLMClient: `complete_json` & `complete_xml` (+`temperature_override`), image data-URLs, circuit breaker (threshold: 2 errors)
│   ├── cv/
│   │   ├── registry.py / loader.py  # DETECTOR_REGISTRY; `.cbz`/`.zip` extraction and directory loading via `natural_sort_key`
│   │   ├── manga_detector.py        # MangaYoloDetector (ONNX): Recursive XY-Cut RTL/LTR (L-shape layouts), inset white-border trimming syncing `focal_points` in Stage 1
│   │   ├── manhwa_slicer.py         # ManhwaStripSlicer: width-relative vertical strip slicing + YOLO `face/body/text` post-pass
│   │   ├── som.py / stage.py        # Set-of-Mark labeling ([P1]..[Pn]) and Stage 1 orchestrator (`run_stage1_ingest`)
│   ├── agents/
│   │   ├── alignment.py             # Shared Semantic & Quote-Pinned Panel Alignment (Stage 3a & 3b): Monotonic DP pre-anchor map, Quote Pinning, monotonicity-aware `verify_beats_alignment` (`quote_mismatch`, `visual_desync`) & `repair_part_panel_alignment_deterministic`
│   │   ├── page_analyzer.py         # Stage 1.5 & 2a (XML): UTF-8 Pre-Pass cast sheet; RAM Visual Verify Code stamping against dropped/shifted pages; CV spread deduplication; Adaptive Batching; `verbatim_quote_vi` & `visual_punch`
│   │   ├── page_index.py            # Stage 2b (XML Summarizer): Full-text alias normalization (`scoped_map`); `first_named_page_id` anti-spoiler reveal; bidirectional spread filtering; Semantic Scene Segmentation; StoryBible (empty default `pronunciation_map`); O(N) `scoped_outline_view`
│   │   ├── scriptwriter.py          # Stage 3a (Decoupled 2-Pass XML): Pass 1 (`ProseWriterAgent`, `temperature=0.72`) writes seamless Video Essay prose across 3–4 Acts from clean CV-free notes; Pass 2 (`PanelAlignerAgent`, `temperature=0.15`) slices beats verbatim using Page Anchor Map + Auto-Retry Gate (`verify_beats_alignment`)
│   │   ├── critic.py                # Stage 3b (Critic V3.7): Deterministic Gate + Semantic Visual-Audio Alignment Gate + Surgical Panel Repair (`CriticPanelRepairer` + Rollback Guard + deterministic safety-net) + Opener/Global Discourse Tic Scanner + Read-Only Auditor + Allowlisted Text Repairer
│   │   └── choreographer.py         # Stage 5: Visual director — word-timestamp sync, 2-beat sub-shot splitting (`Wide -> Focus Face`), LTR/RTL & 9:16 vertical, 7-mood sync
│   ├── audio/
│   │   ├── registry.py / base.py    # TTS_REGISTRY, `apply_pronunciation_glossary`, `beat_cache_key` (including `rate`), anti-aliased Fourier resampling & word timestamp alignment
│   │   ├── edge_tts_backend.py      # Microsoft Edge TTS (Vietnamese) with `WordBoundary` timestamps & parallel synthesis
│   │   ├── vieneu_tts.py            # Local VieNeu TTS backend with automatic Edge-TTS fallback
│   │   ├── music_library.py         # Automated Music & SFX Dataset (`assets/music_library/catalog.json`): downloads CC-BY 4.0 starter pack (Incompetech), DSP analysis (LUFS, BPM, brightness, intro-skip) & custom folder auto-tagging
│   │   ├── music_director.py        # Smart Music Director: selects genre palette from `StoryBible`, groups scene music cues (>=28s), -16 LUFS normalization, 2.5s equal-power crossfades, SFX cooldown & `music_credits.txt` export
│   │   ├── bgm_generator.py / stage.py # BGM orchestration (prefers `music_director` if catalog exists, falls back to 4-bar synth) & Stage 4 driver
│   ├── render/
│   │   ├── registry.py / compositor.py # RENDERER_REGISTRY; OpenCV frame rendering (bokeh, shadow, RTL/LTR diptych, Ken Burns/shake)
│   │   ├── subtitles.py / preview_html.py # `.srt` export, UTF-8 word-wrapped subtitles (9:16/16:9) with `subtitle_font_ratio`, HTML preview with panel thumbnails & audio player
│   │   └── ffmpeg_renderer.py       # Multi-threaded MP4 encoding (VAAPI/libx264), dynamic multi-mood BGM + sidechaincompress ducking
│   ├── exporters/                   # `registry.py`, `srt_exporter.py`, `capcut_draft.py` (full diptych + BGM + subtitles), `html_preview.py`
│   └── downloaders/nyaa.py          # Search and download manga torrents/magnets from Nyaa
└── tests/                           # 8 pytest suites: `test_schemas_and_json.py`, `test_config_and_registry.py`, `test_cv.py`, `test_page_index_and_agents.py`, `test_series_continuity.py`, `test_audio.py`, `test_renderer_and_exporters.py`, `test_downloader.py`
```

### Checkpoint Layout in `jobs/<job_id>/` & Series Memory `series/<series_id>/`
- **Stage 1**: `extracted/` (raw page images), `panels/*.jpg` (cropped panels), `som/*_som.jpg` (Set-of-Mark `[P1]..[Pn]` pages), `01_manifest.json` (`JobManifest`).
- **Stage 1.5 & 2a**: `repo/seed_cast.json` (+`portrait_path`), `repo/cast_sheet.jpg` (dynamic 16-slot character face sheet), `02a_pages/*.json` (per-page Vision cache), `02a_all_pages.json`.
- **Stage 2b**: `repo/notes/scene_*.md` (per-scene Markdown notes with `ar` ratio and `face+text` flags), `repo/tree_index.json` (`PageIndexTree` + `skipped_pages`), `02b_story_bible.json` (`StoryBible`), `repo/MEMORY.md`.
- **Stage 3a & 3b**: `repo/OUTLINE.json` (`1 Scene = 1 Part`), `repo/script/acts/act_*.json` & `repo/script/prose_draft.md` (Pass 1 pure prose), `repo/script/parts/part_*.json` (Pass 2 panel-aligned parts), `repo/script/part_boundaries.json`, `03a_script_draft.json`, `03b_script_final.json`, `03b_critic_report.json`.
- **Stage 4, 5 & 6**: `audio/beats/*.wav`, `04_voiceover.wav`, `04_audio_manifest.json`, `audio/series_music_context.json`, `audio/04b_bgm_dynamic.wav`, `audio/04b_music_cues.json`, `05_timeline.json` (`VideoTimeline`), `06_subtitles.srt`, `06_output.mp4`, `exports/` (+`music_credits.txt`), `telemetry.json`.
- **Series Continuity (`series/<series_id>/`)**: `series_state.json` (`SeriesState`: cross-volume cast, 24+ volume `ledger`, `last_handoff`, `active_plot_threads`, `music_genre_prefs`), `glossary.yaml` (user-editable terminology/pronouns/character overrides between volumes), `portraits/<char_id>.jpg` (persistent Portrait Vault), `compilations/*.mp4|*.srt|*_chapters.txt|*_music_credits.txt` (Arc/Full-Series compiled videos).

---

## 5. Detailed 6-Stage Pipeline Architecture & Data Flow

1. **Stage 1 — CV Ingest (`src/cv/`)**:
   - Extracts `.cbz`/`.zip` using `natural_sort_key`. Automatically selects `ManhwaStripSlicer` (width-relative gutter thresholds + YOLO `face/body/text` post-pass) for webtoons or `MangaYoloDetector` (ONNX YOLOv11 + **Recursive XY-Cut** reading order for L-shape RTL/LTR layouts) for book pages.
   - Trims excess white borders on inset pages and shifts `focal_points` coordinates in Stage 1 (`trim_white_border_with_offset`), then draws Set-of-Mark `[P1]..[Pn]` labels.
2. **Stage 1.5 & 2a — Pre-Pass Cast Sheet & Rolling Vision (`src/agents/page_analyzer.py`)**:
   - **Pre-Vision Blank & Duplicate Spread Filtering**: `_is_blank_or_solid_page()` + `_detect_duplicate_spread_pages()` (bidirectional `[Spread,P1,P2]` and `[P1,P2,Spread]` checks) eliminate blank/redundant pages via pure CV (0 tokens) before calling the Vision LLM.
   - **UTF-8 Pre-Pass & Cross-Volume**: Scans sample pages with dialogue proof (`evidence_quote`), renders `repo/cast_sheet.jpg` via Pillow UTF-8, and optionally inherits `--prev-job`.
   - **Adaptive Panel Batching & Rich Visuals**: Batches dynamically by panel density (`~16–18 panels/call`) with `cast_sheet.jpg` and `RollingSceneState`, extracting `verbatim_quote_vi` + `visual_punch`.
3. **Stage 2b — PageIndex Tree & StoryBible (`src/agents/page_index.py`)**:
   - **Full-Text Scoped Alias Resolution**: Normalizes names across `characters_present`, `page_summary`, `action_summary`, and `dialogue_summary`. Anonymous descriptors stay isolated in `scoped_map` to prevent identity spoilers.
   - **Semantic Scene Segmentation + Note Compression**: Splits scenes at natural boundaries with `narrative_weight` (1–5) & `transition_type` (100% disjoint page coverage verified), compresses `scene_*.md` syntax (eliminating the 24k truncation bug), batches 3 scenes/call with per-scene caching, loads `seed_cast.json`, and generates `pronunciation_map` (`repo/pronunciation.json`).
4. **Stage 3a — Decoupled 2-Pass Scriptwriter (`src/agents/scriptwriter.py`)**:
   - **`1 Scene = 1 Part` Partitioning & Water-Filling**: `OutlinePlannerAgent` creates a semantic outline (`skim`/`steady`/`spotlight`) allocated via `_allocate_part_budgets` (Water-filling + Largest Remainder).
   - **Pass 1 — Pure Storyteller (`ProseWriterAgent`, `temperature=0.72`)**: Groups consecutive parts into **3–4 Acts (`_group_parts_into_acts`)**, feeds clean story notes stripped of CV/panel_id noise (`_strip_cv_metadata_for_prose`), writes fluid, rhythmic **Vietnamese Video Essay prose** (ZERO `panel_id`/XML beat tax, no forced filler tics, still respecting `first_named_page_id` anti-spoiler gates), and saves `repo/script/prose_draft.md`.
   - **Pass 2 — Panel Aligner (`PanelAlignerAgent`, `temperature=0.15`)**: Receives a **Page Anchor Map (`build_alignment_anchors`, Monotonic DP + Quote Pin)** upfront; slices Pass 1 sentences verbatim into `<beat>` tags and binds `1–4 panel_id`s matching exact events/dialogue in monotonic order, backed by an **Auto-Retry Gate (`verify_beats_alignment`)** against phase drift (`quote_mismatch`/`visual_desync`), `_prose_preservation_ratio >= 0.78`, `pin_missing_quote_panels`, and `_sanitize_part_beats`.
5. **Stage 3b — Critic V3.7: Deterministic Gate, Semantic Alignment Gate, Surgical Panel Repair, Global Tic Scanner, Read-Only Auditor & Surgical Text Repair (`src/agents/critic.py`)**:
   - **Tier 1 (Python)**: Checks hard invariants (`grounding`, Part boundaries, backward page jumps, `coverage_gap`, **monotonicity-aware semantic alignment `verify_beats_alignment`**, `spoiler_name`, N-gram `duplication`, AI clichés, `budget > +30%`) plus statistical repetition scanners (`opener_repeat`, **`tic_repeat`** catching filler tics at *any position* outside quoted dialogue, `seam_echo`, and macro colon `:` / staccato sentence density warnings).
   - **Tier 2 & 3a (Visual Surgical Panel Repair `CriticPanelRepairer`)**: Re-aligns `panel_ids` within each Part's allowlist when `panel_sync` / `coverage_gap` issues survive Stage 3a (with a **Rollback Guard** requiring lower error counts and zero page regressions, plus deterministic safety-net `repair_part_panel_alignment_deterministic`) while leaving `narration_vi` 100% untouched.
   - **Tier 2 & 3b (Read-Only `CriticSemanticAuditor` + Surgical Text Repair `CriticSeamPolisher`)**: Reads the entire episode in a single context to flag up to 8 paraphrased repetition or awkward transition beats; edits only **allowlisted `editable_ids`** with a **Per-Beat Rollback Guard**, then populates `tts_text` via `pronunciation_map`.
6. **Stage 4, 5 & 6 — Audio, Smart Music Director, Choreographer & FFmpeg Renderer (`src/audio/`, `src/agents/choreographer.py`, `src/render/`)**:
   - **Audio & Music Dataset**: `manga-recap music fetch/index/list` manages CC-BY/custom tracks at `assets/music_library/catalog.json` (DSP-measured LUFS, BPM, brightness, intro-skip). **`Smart Music Director` (`music_director.py`)** picks a genre palette from `StoryBible.tone_summary`, builds **Music Cues >= 28s** to avoid per-beat track switching, normalizes to `-16 LUFS`, applies **2.5s equal-power crossfades** + real SFX (6s cooldown), and exports `exports/music_credits.txt` (falling back smoothly to a 4-bar synth if no catalog exists).
   - **Choreographer & Renderer**: Word-timestamp subtitle sync, word-wrapped subtitles for `16:9` & `9:16`, 2-beat sub-shot splitting (`Wide -> Focus Face`) on detail-rich long slots, RTL/LTR support in `compositor.py`, HTML preview with `<img>` + `<audio>`, and complete CapCut drafts (diptych + BGM + subtitles).
7. **Multi-Volume Series Continuity Engine (`src/series/`, `src/schemas/series.py`)**:
   - **Portrait Vault & Dynamic Active Cast**: Persists canonical character portraits (`series/<id>/portraits/<char_id>.jpg`), dynamically selects up to 16 active characters for Vol $k$ (prioritizing local cast + `last_handoff` + recent appearances, retiring long-absent/deceased characters), and unlocks `first_named_page_id` for characters revealed in Vol $< k$.
   - **$O(1)$ 3-Tier Memory (`Hot / Warm / Cold`) + Human-Editable `glossary.yaml`**: Loads `last_handoff` (Vol $k-1$), `active_plot_threads` + locked `terminology`/`pronoun_pairs`, and retrieves up to 3 relevant past `ledger` entries for callbacks without token bloat across 24+ volumes.
   - **Dual Continuation Modes (`episodic` vs `seamless`) & Series Compiler**: `episodic` creates standalone episodes with a concise 1–2 sentence bridge from the previous volume; `seamless` suppresses inter-volume intros/outros for `manga-recap series compile` to stitch Arc/Full-Series MP4s with cumulatively shifted `.srt` files, YouTube Chapters, and deduplicated `music_credits.txt`. Locks `music_genre_prefs` and avoids repeating the previous volume's closing track as the next opener.
