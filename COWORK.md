# VERITAS — Cowork Session Handoff

**Purpose:** if this chat stalls or hits context limits, start a fresh Cowork session in this project and feed this file to Claude. It captures the *live* state of in-flight work that isn't yet in CLAUDE.md or git history. CLAUDE.md is the project's long-form truth; this is the short-lived scratchpad for the current push.

**Keep this fresh.** Update after every meaningful step. If it's out of date, the next session will waste time re-discovering state.

---

## Working session started
2026-04-22 (continuation of 0ad3a6c6-38c0-43f2-949d-9fa5422b0771)

## UX backlog (pending)

- **Убрать всю заливку и фоны из таблицы стенограммы.** Стейкхолдер просила «убрать синий цвет» (T1.1 готово — Light Grid Accent 1 → Light Grid). Но в свете её первоначального запроса — она вероятно имела в виду, что ей не нужна **вообще никакая** раскраска / зебра-полосы / заливка в таблице. Только заголовки + содержимое на чистом белом. Сейчас в `_format_court_hearing_docx` (formatter.py) ещё есть стиль таблицы Light Grid (даёт серые границы) и в админ-stenogram возможно есть аналогичная заливка. Свести таблицы к минималистичной white + black-borders без альтернативных строк.
- **Audio player с sync к таймкодам в редакторе транскрипта** — самая большая экономия времени для юристов. Клик по реплике → перематывает плеер. Конкретное предложение: добавить html5 `<audio>` элемент в TranscriptPage с обработчиком на каждом сегменте.
- **Bulk speaker rename** — сейчас можно переименовать только через контекстный rename. Для случая «pyannote перепутал двух спикеров на всю запись» нужен drag-n-drop или быстрый bulk-режим.
- **Highlight low-confidence слов** прямо в тексте — у нас есть `attribution_confidence` на уровне сегмента, но не на уровне слов. Использовать word-level confidence из ASR (есть в `WordInfo`) и подсвечивать конкретные слова, которые модель не уверена.

## Last updated
2026-04-28 (third pass) — VRAM resilience sprint: pre-flight + dashboard + retry.

User hit "Insufficient VRAM" mid-pipeline (another GPU app held 22.8 GB on a 24 GB card; only 1.2 GB free). Pipeline failed after 1-2 min with cryptic error and no recovery. Sprint scope: surface the issue early, show the user what's happening, let them retry without re-uploading.

**Backend:**
- `backend/app/routes/meetings.py` upload endpoint now does a pre-flight VRAM check via `VRAMManager.get_status()`. If free < 4 GB, returns 503 with a Russian message naming likely culprit apps (LM Studio, Stable Diffusion, ChatGPT desktop, фоновые игры). Same check on the new retry endpoint.
- New endpoint `POST /api/meetings/{job_id}/retry`: reuses an existing failed job, resets state to UPLOADED + clears error, increments retry_count, re-triggers background_process_meeting. Refuses to retry if job is not in error state, or if another job is processing, or if VRAM is still low.

**Frontend:**
- New component `frontend/src/components/GpuStatus.tsx` — polls `/api/system/gpu` every N sec, shows a coloured progress bar (green/amber/red by free GB) and a "Обновить" button. Used as a reusable building block.
- `UploadPage.tsx`: live VRAM polling, button gated by `vramLow = free < 4 GB`, `<GpuStatus hideWhenHealthy />` shown above the submit button so it only appears when memory is tight or unavailable. Tooltip on disabled button explains why.
- `ProcessingPage.tsx`: extracted error rendering into a new `ErrorBlock` component. Detects VRAM-style errors via regex (`/vram|видеопам|insufficient/i` or HTTP 503) and renders a special panel with: explainer text, embedded `<GpuStatus refreshSec={3} />` for live monitoring, list of likely culprit apps, "Повторить" button calling `retryMeeting()`, "Загрузить заново" fallback, and a collapsible technical details section. Generic errors get a simpler retry+upload-again pair.
- `client.ts`: new `retryMeeting(jobId)` helper.

**What this fixes for users**: instead of waiting 1-2 min for a cryptic mid-pipeline error and being stuck with no recovery path, they now see GPU memory state on the upload page (button greys out automatically), get a clear explanation if it fails mid-run, and can hit "Повторить" to resume without re-uploading audio after freeing memory.

## Last updated
2026-04-28 (second pass) — Quality defaults updated for legal use case.

After the GigaAM-vs-WhisperX comparison numbers came back (25.7% vs 32.5% WER on court_hearing_129), shifted defaults to favour quality for the most-error-sensitive use case:

1. **LLM post-correction (T3.4) now ON by default for court_hearing.** Backend logic in `_async_process_meeting` on the orchestrator: if `enable_llm_correction` not explicitly set by caller AND meeting_type == COURT_HEARING → enable it. UI-side, the upload page's checkbox auto-checks (and the "Дополнительные опции" panel auto-expands) when user picks «Судебное заседание». User can still uncheck if they want speed. Other meeting types: still opt-in.
2. **WhisperX marked «(экспериментально)» in UI.** Description updated to cite the empirical WER deficit ("на бенчмарке court_hearing_129 показал WER 32.5% против 25.7% у GigaAM, на 7 п.п. хуже"). Engine code stays in place; just clearer signal so non-technical users don't accidentally pick the worse engine.
3. **GigaAM description bumped** to explicitly say it's recommended for all Russian-language meetings (was just stating the training stats).

These are presentation and default-shift changes only — no new code path, just steering the user toward what we now have evidence is best.

## Last updated
2026-04-28 — ASR comparison findings + filter regression fix.

User ran `COMPARE_ASR_ENGINES.bat` on court_hearing_129. Results:
- GigaAM: WER 25.67%, CER 18.51%, 218 segments, 6,811 words (~ matches gold's 6,817)
- WhisperX: WER 32.51%, CER 23.92%, 190 segments, 5,810 words (15% under-transcription, 1,238 deletions vs GigaAM's 517)
- **GigaAM wins decisively on Russian legal speech.** WhisperX's wav2vec2 alignment isn't worth its 7-pp WER deficit on this content. Recommendation: keep GigaAM as default, treat WhisperX as a niche optional engine (or shelve it entirely).

User flagged TWO regressions caused by our own postprocessor:
1. **T2.4 hallucination filter dropped a real witness number.** Gold turn 13 was «22, 14, вы оплачиваете». GigaAM transcribed as «22:14 вы оплачиваете?» (close enough, 21 chars). Our `_TIMECODE_ARTIFACT_RE` matched H:MM at start, segment <30 chars → **deleted entirely**. Real legal speakers say times like «22:14» or «3:30» constantly; old regex was too loose.
2. **`merge_short_segments` 3.0s threshold was over-aggressive** on WhisperX's micro-segments (635 → 190, 70% reduction).

Fixes shipped:
- `_TIMECODE_ARTIFACT_RE` tightened from `H:MM(:SS)?` to `H:MM:SS` (3 colon-parts required). Whisper hallucinations on tishina nearly always have seconds (YouTube subtitle artifact); real spoken times don't. Should kill the false positive without losing the protection.
- `filter_hallucination_artifacts()` now logs at INFO level whenever it drops a segment, with the exact text. Makes future false positives visible for audit.
- `merge_short_segments` `max_turn_gap_s` lowered 3.0s → 2.0s. Preserves natural same-speaker continuations while reducing wall-of-text merges.

Next: re-run comparison after fixes to confirm the «22:14» content reappears in GigaAM output.

## Last updated
2026-04-23 (thirteenth pass) — ASR engine comparison harness.

User wants to A/B GigaAM vs WhisperX against the court_hearing_129 gold to decide which is better for their content.

Built:
- **`benchmarks/scripts/run_admin_e2e.py`** gained `--engine` (force a specific ASR engine, overrides config), `--archive-prefix` (separate folder per comparison run so they don't collide), and `--skip-summarization` (cuts 5-10 min/run; LLM output isn't part of ASR comparison).
- **`benchmarks/scripts/compare_asr_engines.py`** (new): runs the pipeline once per engine, scores each engine's concatenated transcript against `benchmarks/gold/court_hearing_129.jsonl` via `score_wer.py` (WER + CER + sub/del/ins counts), generates HTML report at `benchmarks/reports/asr_compare_<stamp>.html` — 3-column side-by-side (Gold | GigaAM | WhisperX) with stats panel at top, color-coded WER (green <10%, amber 10-30%, red >30%). Plus JSON summary for CI.
- **`COMPARE_ASR_ENGINES.bat`** (new): one-click. Pre-flight checks venv, gold file, audio, WhisperX cached models. Calls compare_asr_engines.py. Opens latest HTML in browser when done.

User flow: double-click `COMPARE_ASR_ENGINES.bat` → wait 30-60 min → browser opens with the comparison report.

## Last updated
2026-04-23 (twelfth pass) — WhisperX install hardening + ASR load-fallback chain.

User installed WhisperX, picked it in UI, pipeline failed at runtime with `requests.exceptions.SSLError` trying to fetch `Systran/faster-whisper-large-v3` from HuggingFace. Two issues:
1. `INSTALL_WHISPERX.bat` previously only ran `pip install` — didn't pre-download the ~4.5 GB WhisperX models. The launcher then sets `HF_HUB_OFFLINE=1` for security, blocking runtime downloads. Result: install passed, runtime failed.
2. When WhisperX fails to load mid-pipeline, the whole job died.

Fixes:
- **`INSTALL_WHISPERX.bat`** now has step 4/4: temporarily unsets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` and pre-downloads both Whisper large-v3 (~3 GB) and `jonatasgrosman/wav2vec2-large-xlsr-53-russian` (~1.5 GB) into the HF cache. After install, the runtime works fully offline against the cached models. Clear failure messages if the network is bad: try again / use `HF_ENDPOINT=https://hf-mirror.com` / different network / fall back to GigaAM.
- **Orchestrator ASR load-fallback chain**: in `process_meeting`, when the chosen engine's `.load()` raises, log the error and try the next engine in the fallback chain. For WhisperX, chain is `whisperx → whisper → gigaam`. User's job completes with whichever engine successfully loads, with a `WARNING` log explaining the swap. Was: any load failure killed the entire meeting.

User next step: re-run `INSTALL_WHISPERX.bat` (download step is the new addition, will populate the cache). After it succeeds, restart backend and pick WhisperX in the UI again.

## Last updated
2026-04-23 (eleventh pass) — Backend port moved off 8000 to 8765.

User runs three projects simultaneously; one of them already binds 8000 and VERITAS couldn't start. Default port now **8765** everywhere:
- `backend/app/config.py` ServerConfig.port = 8765
- `config/settings.yaml` server.port = 8765
- `backend/app/main.py` falls back to `EPAM_BACKEND_PORT or 8765`
- `frontend/vite.config.ts` reads `process.env.EPAM_BACKEND_PORT` (default 8765) for `/api` and `/ws` proxy
- `START_VERITAS_MVP.bat`:
  - Pre-flight `netstat` check that the chosen port is actually free; clear error message + pause if not (with instructions for either stopping the colliding process or setting `EPAM_BACKEND_PORT=8766` in `.env`)
  - Sets the env var, passes it to both backend (via `--port`) and frontend (via env)
  - All "Backend is ready" / "Backend API URL" messages reference `%EPAM_BACKEND_PORT%`
- `LoginPage.tsx` "Backend is not reachable" hint mentions 8765

Override path for power users: edit `.env` and add `EPAM_BACKEND_PORT=8766` (or any free port). Launcher reads it before starting both processes.

## Last updated
2026-04-23 (tenth pass) — UI toggles for Tier 3 opt-in features.

User flagged that "try EPAM_…=true" instructions are useless for non-developers. Surfaced T3.4/T3.5/T3.6 as buttons:

- **`INSTALL_WHISPERX.bat`** in project root: double-click installs WhisperX in the venv, verifies import, verifies CUDA didn't break. Self-contained, no CLI.
- **`/api/system/features` endpoint** reports each opt-in feature's `available` (deps installed?) and `default_enabled` (config flag) status. Used by frontend to gate WhisperX option behind whether `import whisperx` works.
- **Per-meeting form fields** on POST /api/meetings/upload: `enable_llm_correction`, `enable_polish_pass`, `asr_engine_override`. Routed through orchestrator → summarization, override config defaults for THIS job only. Stored in `orchestrator._job_overrides[job_id]` between upload-handler and background-task; cleaned up after.
- **UploadPage advanced section** (collapsible "Дополнительные опции" with chevron icon): two checkboxes (LLM-коррекция, Полировка форматирования) + ASR engine radio (Авто / GigaAM / Whisper / WhisperX). Polish-pass disabled unless meeting type = administrative. WhisperX disabled with explainer ("Запустите INSTALL_WHISPERX.bat") when not installed.
- All overrides default to "use server config" (sent only when explicitly set to true).

User-visible result: open the upload page, expand "Дополнительные опции", check what you want to try, upload. No env vars, no `pip install` commands.

## Last updated
2026-04-23 (ninth pass) — Tier 3 complete (six tasks shipped in one sprint).

**T3.1 Speaker-count buckets (UI)** — replaces numeric expected-speakers dropdown with named buckets per IT recommendation: Auto-detect / 2 speakers / 3-5 / 6-10 / 11-15 / 16-20 / More than 20. Frontend resolves to (min, max) via `resolveSpeakerBucket()`; backend `expected_speakers_min` + `expected_speakers_max` form fields plumbed through `background_process_meeting → process_meeting → diarization`. Legacy `expected_speakers` single field still supported.

**T3.2 Ollama retry-with-backoff** — `summarization._generate()` wrapped with 3-attempt retry (1s/3s/9s exponential). Retries 5xx + 408 + ConnectError + TimeoutException; bails fast on 4xx (programming errors). Same final error messages preserved for compat.

**T3.3 Flat 9-section admin format** — drops department grouping. New `AdministrativeProtocol` shape: `meeting_date / meeting_goal / summary / participants / topics / decisions[] / tasks[] / open_questions[] / risks[] / turns`. Each item carries `text / speaker / evidence`; tasks use `owner / deadline` instead of `speaker`. New `FlatProtocolItem` schema. Prompt rewritten to match IT's 7-key schema verbatim (with our extensions for `meeting_date` + `evidence`). Verifier rewritten for new shape — only deletes, never edits. Parser simplified (no slug canonicalisation). Formatter dispatches on which fields are populated: legacy `_format_administrative_docx_legacy` for old protocols on disk, new `_format_administrative_docx_flat` for everything else. New helpers `_render_flat_items_with_speaker` (bullet list + speaker italic) and `_render_flat_tasks_table` (3-col Задача / Исполнитель / Срок). Speaker rename handler updated to update flat-schema speaker/owner fields too.

**T3.4 LLM post-correction pass (opt-in)** — new pipeline stage `backend/core/transcript_corrector.py` between post-processing and summarization. Runs Gemma over batches of ~4000 chars with strict "fix only obvious ASR errors, never paraphrase" prompt. Validation gates: turn count must match, speaker must match, per-turn Jaccard token overlap >= 0.70 (paraphrase guard reverts individual turns). Reuses summarization engine's Ollama client. Default OFF (`postprocessing.llm_correction: false`) — flip to true per run via env var while gathering data. Wired into orchestrator after `postprocess_transcript`.

**T3.5 Polish pass on admin protocol** — new `_polish_admin_protocol()` in summarization.py. Runs after parse_response, only for `MeetingType.ADMINISTRATIVE`. One Gemma call with format-only-edits prompt, validates length-preserving + speaker/owner/evidence-preserving invariants before accepting. Default OFF (`summarization.polish_pass: false`). Stenogram turns stripped from prompt input and restored after to keep prompt small and protect verbatim transcript.

**T3.6 WhisperX ASR engine (pilot)** — new `backend/engine/whisperx_asr.py`. Optional dependency: `try: import whisperx`; if missing, engine raises ImportError on `__init__` and orchestrator falls back to faster-whisper. NEVER auto-routed — only used when `config.asr.engine = "whisperx"` is set explicitly. Loads Whisper large-v3 + `jonatasgrosman/wav2vec2-large-xlsr-53-russian` for forced word-level alignment. Returns TranscriptionSegment with phoneme-aligned word timestamps that are tighter than vanilla Whisper's chunk-level output. To install in venv: `venv\Scripts\python.exe -m pip install whisperx`.

To take effect (single restart):
1. Restart backend (close VERITAS Backend window, re-launch START_VERITAS_MVP.bat).
2. Refresh frontend tab (Vite HMR for the speaker bucket UI).
3. Optional T3.4/T3.5 enable: set `EPAM_POSTPROCESSING_LLM_CORRECTION=true` and/or `EPAM_SUMMARIZATION_POLISH_PASS=true` in `.env` before launching.
4. Optional T3.6 enable: `pip install whisperx` in the venv, then `EPAM_ASR_ENGINE=whisperx`.

Pre-Tier-3 changes still in place: text2num disabled by default, MP4 intake, all Tier 1/2 fixes, speaker-rename DOCX-blanking bug fix.

## Last updated
2026-04-23 (eighth pass) — text2num revert + MP4 intake.

User reviewed protocol (14).docx (post-Tier-2). Two text2num regressions confirmed:
- "Двадцать пятого" → "20 пятого" (text2num converts cardinal portion of an ordinal but doesn't know the genitive suffix; date phrases break)
- Standalone narrative "миллион" / "миллионов" → "1 000 000" inappropriately ("5 с половиной 1 000 000")
- 23+ such artifacts in 60 min of audio; reference baseline had zero.

Fix: `normalize_numbers` default flipped to **false** in `config.py` and `settings.yaml`. text2num wrapper code retained for a future, narrowly-scoped normalizer that only fires on `X (тысяч|миллионов) (рублей|долларов)` patterns with explicit currency anchors. Will revisit in Tier 3.

MP4 / video intake added (was: audio-only):
- `backend/app/routes/meetings.py` upload validator now accepts `.mp4 .mov .avi .mkv .webm` in addition to audio.
- `backend/core/audio.py` raises a clear error if a video file is uploaded but ffmpeg isn't available (otherwise torchaudio fallback would crash cryptically).
- `frontend/src/pages/UploadPage.tsx` `ALLOWED_AUDIO_EXT` extended; help text and drop-zone caption updated.
- ffmpeg already strips the audio track from any video container — no preprocessor changes needed beyond the validator update.

Tier 2 changes still in place (chunk-overlap recovery, pyannote overlap detection, confidence UI, hallucination filter) and confirmed working. Timestamps `[HH:MM:SS]` confirmed on all 218 turns.

## Last updated
2026-04-23 (seventh pass) — Tier 2 quality sprint shipped. Comparison of post-T1 protocol (13).docx vs stakeholder reference confirmed T1 delivered cosmetic + segmentation wins (~50-60% edits remaining vs 12.8/min baseline) but left model-level errors untouched. Tier 2 attacks segment-boundary recovery, overlap detection, hallucination cleanup, and editor UX.

- **T2.1** [DONE] `gigaam_asr.py` — Silero-VAD chunks now extended by `tail_overlap_s=1.5s` audio for better right-context. New `_dedupe_chunk_overlap()` runs after transcription: word-level suffix-prefix matching between adjacent chunks (case-insensitive, punct-stripped, requires >=2 matching words to fire), strips the duplicate prefix from chunk N+1's text + word list. Logs word-count delta.
- **T2.2** [DONE] `models.py` `DiarizationSegment.is_overlap` field. `pyannote_diarization.py` new `_mark_overlaps()` sweep-line marks segments that share >=50ms with another speaker's segment. `aligner.py` accumulates `overlap_time` per ASR window and caps `attribution_confidence` at 0.55 when overlap-share >= 10% — so overlap regions automatically land in the "contested" UI bucket.
- **T2.3** [DONE] `TranscriptPage.tsx` color-codes segments by attribution_confidence (red <0.6, amber <0.8, neutral >=0.8). New "Только к проверке" filter button hides high-confidence rows. Header strip shows "{N} из {M} реплик помечены к проверке" stat. Per-segment "проверка"/"спорно" badge for low-confidence. `AlignedSegment` TS type extended with `attribution_confidence`.
- **T2.4** [DONE] `postprocessor.py` `filter_hallucination_artifacts()` strips short segments matching Whisper hallucination patterns (timecode-like artifacts at chunk boundaries, YouTube-leftover Russian/English phrases). Conservative: only fires on segments <=30 chars dominated by the pattern. Wired via `EPAM_POSTPROCESSING_FILTER_HALLUCINATIONS` (default true).

To activate (no new pip dep needed for T2):
1. Restart backend (close VERITAS Backend window, re-launch START_VERITAS_MVP.bat).
2. Refresh frontend (Vite picks up the TS changes).
3. Re-run on the same court_hearing audio. Expected: cleaner clause boundaries (T2.1), overlap regions amber/red in UI (T2.2+T2.3), "22:14 ..." artifacts gone (T2.4).

Tier 1 changes still in place (require text2num install in venv if not done):
- formatter blue→grey, [HH:MM:SS] timestamps, legal initial_prompt, VAD retune, Russian number normalisation.

Pre-Tier-1 fixes:
- speaker-rename DOCX-blanking bug fixed; logging at INFO; english_threshold 0.50; admin schema gained `turns`; admin DOCX renders СТЕНОГРАММА.

**Next: Tier 3 candidates** (proposed during Tier 2 sprint review):
- T3.1 LLM post-correction pass over transcript turns (highest-impact fix for word-level ASR errors like "прощения" → "прошу")
- T3.2 Source separation on overlap regions (Sepformer/MossFormer2 — actually transcribes overlapping speakers)
- T3.3 Voice enrollment seed (Block 7 starter) — short voice samples for known speakers
- T3.4 Diff-harness regression test (uses stakeholder track-changes file as ground truth, surfaces edit-rate as a number per release)

## Last updated
2026-04-23 (sixth pass) — Tier 1 quality sprint shipped (5 changes targeting stakeholder feedback on court_hearing_129).

Stakeholder review of 10-min court hearing edit (128 insertions / 0 deletions; 76 phrase completions / 15 speaker labels / 15 word fragments / 11 punct / 6 numbers / 5 caps) drove these:

- **T1.1** [DONE] formatter.py — replaced `Light Grid Accent 1` (Word's theme-blue) with `Light Grid` (neutral grey) on all three stenogram tables (court, admin, generic).
- **T1.2** [DONE] schemas.py CourtHearingTurn gained `start_s: Optional[float]`; court_hearing.parse_response and administrative.parse_response capture seg.start; formatter renders `[HH:MM:SS]` (or `[MM:SS]` for hearings <1h) in light-grey 8-9pt next to the speaker label. Both court table and admin stenogram paragraphs.
- **T1.3** [DONE] config.asr.initial_prompt expanded with full legal-domain vocabulary (истец/ответчик/ходатайство/апелляция/кассация/etc., currency words). Wired into faster-whisper transcribe_kwargs (`whisper_asr.py`) and HFWhisper via `processor.get_prompt_ids()` → `prompt_ids` in `model.generate()` (`hf_whisper_asr.py`). GigaAM v3 has no equivalent API in the gigaam pip package, so it skips this.
- **T1.4** [DONE] VAD retune: `vad_threshold 0.5→0.4`, `vad_min_silence_ms 100→350` in `config.py` and `settings.yaml`. Wired through to GigaAM longform's `_run_silero_vad` (was hardcoded `threshold=0.5, min_silence_duration_ms=100`); now reads from config.
- **T1.5** [DONE] postprocessor `normalize_russian_numbers()` via `text2num.alpha2digit("ru", relaxed=True)`, plus NBSP thousands-separator post-pass for legal-style readability ("6 407 000 рублей"). New `text2num>=2.4,<3.0` dep in requirements.txt. New `EPAM_POSTPROCESSING_NORMALIZE_NUMBERS` config (default true), wired via orchestrator. Russian-only; auto-skipped on non-RU language.

To take effect:
1. `pip install text2num>=2.4` in the venv (or full REBUILD_VENV).
2. Restart backend (close VERITAS Backend window, re-launch START_VERITAS_MVP.bat).
3. Re-run on the same court_hearing audio. Expected improvements: no blue grid, [HH:MM:SS] before each turn, fewer mid-clause cuts (T1.4 should reduce most of the 76 phrase-completion edits), digits where amounts/years were spelled out.

Pre-Tier-1 fixes still in place:
- formatter blue→grey, but the DOCX-blanking bug from the rename handler was the actual root cause of the previous "blank protocol" report; that fix (`rename_speaker` now dispatches parsing on `meeting_type` to AdministrativeProtocol/CourtHearingProtocol/etc.) is shipped.
- Logging fix (basicConfig at INFO in main.py) so future diagnostics aren't blind.
- english_threshold raised 0.20 → 0.50.
- AdministrativeProtocol gained `turns`; admin DOCX renders СТЕНОГРАММА.

Pending:
- Tier 2 (next 1–2 weeks): chunk-overlap recovery, pyannote overlap-aware diarization, confidence-based UI highlighting.
- Task #6 (flat 9-section admin format): deprioritised after rename-bug fix likely solves the original "blank" complaint. User should re-evaluate the next admin DOCX before deciding.

Will look like:
2026-04-23 (fifth pass) — REAL root cause found: SPEAKER RENAME HANDLER corrupted the DOCX after regeneration, not ASR or LLM.

User uploaded the job artifacts (aligned.json, aligned_raw.json, protocol JSON, DOCX). Agent inspection found:
1. ASR was actually GigaAM (not Whisper as I mis-diagnosed). 849 clean segments, 0.95 avg confidence, native punctuation+capitalization. The 2289-2304s hallucination alarm was a minor artifact, not a systemic failure.
2. LLM produced a fully populated protocol JSON: 615-char summary, 17 participants (all 8 checkbox attendees + 9 LLM-extracted), 2 decisions, 3 tasks, 3 open_issues distributed across departments. Real content everywhere.
3. **The DOCX was blank because `rename_speaker` in routes/meetings.py (line 584-588) re-parsed the on-disk JSON as legacy `MeetingProtocol` class, which doesn't know about admin's `{meeting_type, payload: {...}}` wrapper schema. Pydantic silently discarded all unknown keys → built empty MeetingProtocol → formatter dispatched to `_format_generic_docx` → blank output.** User's rename action immediately before DOCX download erased everything the LLM produced.

Fix landed (2026-04-23 fifth pass):
- `rename_speaker` now detects the on-disk JSON shape (wrapped `{meeting_type, payload}` vs legacy flat), parses into the RIGHT Pydantic class (AdministrativeProtocol / CourtHearingProtocol / ClientMeetingProtocol / InterviewProtocol / MeetingProtocol), and updates speaker fields in the right places per schema (payload.participants strings, payload.turns[].speaker, department items[].speaker).
- Resummarize path was already clean (live Pydantic, no JSON round-trip).

Also from this session (4th pass, before root cause was clear):
- `main.py` — added `logging.basicConfig(level=INFO, force=True, ...)` so `backend.*` INFO logs surface to the backend cmd window (still useful regardless — `EPAM_LOG_LEVEL` env override).
- `config.py` + `settings.yaml` — `asr.english_threshold` raised 0.20 → 0.50 (still a sensible hygiene change).
- `schemas.py` — `AdministrativeProtocol.turns` field (enables СТЕНОГРАММА rendering).
- `administrative.py` — context-fallback parser for participants, softened prompt summary rules.
- `formatter.py` — СТЕНОГРАММА section at end of admin DOCX.
- `summarization.py` — `meeting_context` plumbed to parse_response.

**Still pending (task #6):** user wants flat 9-section admin format matching the "decent" baseline. Queued but deprioritized now that the actual DOCX-blanking bug is fixed; the current department-grouped output likely renders fine post-fix. User should re-test first.

---

## Current block

**Block 6: SOTA Model Upgrade (RTX 3090) — LLM prompt hardening in progress**
Pipeline infrastructure is green. Focus shifted to summarization output quality.

## Current plan

1. **[DONE]** Rip DiariZen code from repo (CC BY-NC licensing blocks commercial rollout). CLAUDE.md reflects this.
2. **[DONE]** MeetingType → ASR dispatch. court_hearing → GigaAM, everything else → hf-whisper (antony66). Implemented in `backend/core/orchestrator.py` (`_resolve_asr_engine`, `_instantiate_asr`).
3. **[DONE]** Venv rebuilt. Nine dependency traps resolved, documented in memory.
4. **[DONE]** E2E smoke test on Админ 13-04-2026. 9.6 min wall time, AdministrativeProtocol generated (quality found lacking — see #5).
5. **[DONE, 2026-04-23]** Switch default LLM from T-Pro 2.0 to Gemma 4 26B (`config/settings.yaml` → `ollama_model: "gemma4:26b"`). Rewrite `backend/engine/protocols/administrative.py` prompt using EPAM IT team's Russian-schema production prompt as base. Strengthen verification prompt (delete-only rules, evidence checking). Parser normalizes IT's "нет" convention. Launcher `START_VERITAS_MVP.bat` checks `gemma4:26b` is pulled in Ollama.
6. **[NEXT]** User launches `START_VERITAS_MVP.bat`, uploads a court-hearing audio file via the UI, observes the full pipeline end-to-end from browser. Verifies: (a) UI stack starts cleanly, (b) meeting-type selector picks court_hearing → routes to GigaAM, (c) output quality improved vs. T-Pro run.
7. **[PENDING]** Review Админ DOCX from 2026-04-22 run in detail, catalogue specific failure modes to inform further prompt tuning.
8. **[PENDING]** Debug court_hearing E2E script failure from 2026-04-22 (blocked on user pasting error).

---

## Active blocker

**None.** Venv is green. All nine dependency traps from the 2026-04-22 install cascade are fixed and documented in memory `feedback_pyannote_reinstall_trap.md`. Running count:

1. **[fixed earlier]** huggingface-hub auto-upgrades to 1.11+ (breaks transformers). Pin `"huggingface-hub>=0.34.0,<1.0"`.
2. **[fixed earlier]** speechbrain too old for torchaudio 2.8 API. Pin `"speechbrain>=1.0.3"`.
3. **[fixed earlier]** `lightning` + `pytorch-lightning` both installed → torch kernel registration conflict. Use `lightning` only.
4. **[fixed earlier]** `--force-reinstall` on pyannote silently replaces CUDA torch with CPU torch. Re-install torch trio with `--no-deps --index-url pytorch cu126`.
5. **[fixed + VERIFIED]** `lightning>=2.0,<3.0` too wide → pip backtracks to Lightning 2.3.3, violates pyannote-audio 4.0's transitive `>=2.4`. Fixed: tightened to `lightning>=2.4,<3.0`.
6. **[fixed + VERIFIED]** `torchmetrics 1.9.0` collides with lightning 2.4.0 via `lightning-utilities`. Fixed: added `torchmetrics>=1.4,<1.9`.
7. **[fixed + VERIFIED]** PyPI `gigaam 0.1.0` pins `torch<=2.5.1`, blocks resolver. Fixed: removed `gigaam` line from requirements.txt; GitHub install handles it at Step 5 with `--no-deps`.
8. **[fixed + VERIFIED]** `torchcodec` pulled in transitively by pyannote.audio 4.0.4. DLLs fail on Windows + torch 2.8. Fixed: REBUILD_VENV.bat uninstalls torchcodec in Step 4.5; `pyannote_diarization.py` already pre-loads audio as `{"waveform", "sample_rate"}` dict.
9. **[fixed + VERIFIED]** GigaAM GitHub `--no-deps` install skips hydra-core → `import gigaam` raises `ModuleNotFoundError: No module named 'hydra'`. Fixed: added `hydra-core>=1.3,<2.0` to requirements.txt (pulls omegaconf transitively).

Benign warnings observed on verified venv:
- `torchcodec is not installed correctly — use preloaded waveform dict` (informational; engine already does this).
- `gigaam 0.1.0 requires onnx==1.19.* / onnxruntime==1.23.*` (unused ONNX export path, leave alone).

---

## Verification gate (run after venv rebuild)

Before moving to the E2E test, confirm torch survived the install:

```
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.__version__)"
```

Expected: `cuda: True 2.8.0+cu126`.
If `+cpu`: reinstall torch trio:
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
```

---

## Smoke test plan (step 4)

From the VERITAS 2 directory with venv activated:

```
# Administrative meeting type → hf-whisper (antony66) ASR
python benchmarks\scripts\run_admin_e2e.py

# Court hearing → GigaAM ASR
python benchmarks\scripts\run_admin_e2e.py --meeting-type court_hearing
```

**What to watch in the log:**
- `"Meeting <id>: type=<value>, ASR engine=<engine>"` — confirms `_resolve_asr_engine(meeting_type)` fired with the right input. If you see `ASR engine=auto` the config override didn't take.
- 3-stage VRAM lifecycle (ASR load/unload → diarization load/unload → Ollama summarization — Ollama manages its own VRAM and shouldn't appear in VRAMManager logs).
- Clean protocol generation at the end (DOCX + JSON in `data/archive/`).
- No `(no CUDA device)` warnings → would mean torch fell back to CPU, abort.

---

## Files touched in this session

- `backend/app/config.py` — `asr.engine` default `"gigaam"` → `"auto"`; DiariZen fields removed from DiarizationConfig.
- `backend/core/orchestrator.py` — added `_resolve_asr_engine`, `_instantiate_asr`; refactored `_ensure_engines_loaded` to accept `meeting_type`; DiariZen branch removed; logs ASR engine at upload time.
- `config/settings.yaml` — `asr.engine: "auto"` with routing comment; DiariZen entries removed.
- `requirements.txt` — `lightning>=2.4,<3.0` (was `>=2.0`), new `torchmetrics>=1.4,<1.9` cap.
- `benchmarks/scripts/build_admin_compare_html.py:521` — DiariZen example → pyannote 4.0.
- `benchmarks/README.md:95` — DiariZen row removed from table.

**Deleted:**
- `backend/engine/diarizen_diarization.py`
- `DIARIZEN_SETUP.md`
- `RUN_DIARIZEN_COMPARE.bat`
- `RUN_PYANNOTE_VS_DIARIZEN.bat`

**Memory updated:**
- `feedback_pyannote_reinstall_trap.md` — traps #5 and #6 added with dates.
- `project_diarization_licensing.md` — DiariZen CC BY-NC note (from earlier).

---

## Not-yet-done from the bigger backlog (deferred after smoke test passes)

- **Block 7 Phase 1F:** Frontend meeting-type picker on UploadPage (dropdown for MeetingType enum).
- **Block 7 Phase 1G:** Wall-of-text paragraph-break postprocessor — parked from 2026-04-21 Phase A+ session. Polished turns >~60s should get soft paragraph breaks at sentence boundaries.
- **Three-column alignment output** — analyze and deliver findings.
- **Unicode dash normalization** — do *before* filler removal so `—`/`–`/`-` variants all get the same treatment.
- **Post-processing enhancements** — several smaller items in the tracker.
- **Subprocess isolation for VRAM phases** (optional optimization, not required for correctness).

---

## Gotchas / non-obvious constraints

- **DiariZen is CC BY-NC.** Do not re-add, even for "just a benchmark" — user confirmed 2026-04-22 current work is a tech eval, next rollout is commercial product. Decision is final.
- **No torchcodec on Windows + torch 2.8.** Uninstall on sight (`pip uninstall torchcodec -y`). pyannote 4.0 imports it silently but crashes at runtime. GigaAM longform uses Silero VAD instead; pyannote diarization pre-loads audio via torchaudio into `{"waveform", "sample_rate"}` dict.
- **Single-pass summarization.** User explicit: no chunking even for long meetings. T-Pro 2.0 Q4_K_M has 32K context, Gemma 4 has 128K — the whole transcript goes in one call.
- **Ollama manages its own VRAM.** Summarization stage must NOT register/unregister with VRAMManager; it calls Ollama over HTTP and Ollama handles GPU lifecycle.
- **All code in English.** No Cyrillic in source files. Comments, docstrings, variable names all ASCII.
- **Security is non-negotiable.** Law firm data. No public code. No untrusted deps.

---

## How to resume this session in a fresh chat

1. Open a new Cowork chat in VERITAS 2.
2. Share this file — `COWORK.md` in the project root.
3. Tell Claude: "We were mid-rebuild — read COWORK.md, confirm my venv status, pick up from there."
4. Claude will check the **Active blocker** section, ask you whether REBUILD_VENV.bat finished cleanly, and proceed to the verification gate / smoke test as appropriate.

The long-form project history lives in `CLAUDE.md` at the project root. The memory file `feedback_pyannote_reinstall_trap.md` (in Claude's persistent memory) has the full dependency-cascade playbook.
