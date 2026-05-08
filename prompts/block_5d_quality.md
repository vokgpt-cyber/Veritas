# Block 5d: ML Quality Upgrade — Coding Session Prompt

Read CLAUDE.md first — the full plan is in "Block 5d: ML Quality Upgrade". Follow all Session Instructions.

## Context

Real-world E2E test results (2026-04-07, on-premise, no internet):
- 11 real speakers → only 7 detected (under-segmentation)
- Transcription and summarization quality poor on this complex meeting
- For 2-3 speaker meetings the current pipeline works okay
- Quality collapses with many speakers (5+)
- All models are already downloaded and cached locally

## Architecture Change: Pipeline Order (CRITICAL — do this first)

**Current order:** Preprocess → ASR → Diarization → Align → Summarize
**New order:**    Preprocess → Diarization → [Select Profile] → ASR → Align → Summarize

**Why:**
- ASR and diarization are INDEPENDENT — both work from raw audio
- Diarization is faster (~1-2 min) and lighter (~300MB VRAM) than ASR (~3GB, 5-10 min)
- Running diarization FIRST on FULL audio gives EXACT speaker count, not a sample guess
- That count drives profile selection for ASR and summarization

**Current hidden dependency to remove:**
The orchestrator (line ~218) extracts speech_segments from ASR output and passes them to diarization as VAD hints. This is an artificial dependency. Silero VAD (`backend/engine/vad.py`) already exists but is DEAD CODE — never used in the pipeline. When flipping the order, use Silero VAD to provide speech activity detection for diarization instead of ASR boundaries. This makes the engines truly independent.

**Implementation in `orchestrator.py`:**
1. After preprocessing, run Silero VAD on the preprocessed audio to get speech segments
2. Pass VAD segments to diarization (instead of ASR output)
3. Run diarization on full audio → get speaker count
4. Select profile based on speaker count (<=4 → SMALL, >=5 → LARGE)
5. Apply profile settings to ASR config before loading Whisper
6. Run ASR with profile-tuned parameters
7. Continue with alignment → post-processing → summarization (with profile-tuned prompts)
8. Update progress percentages: diarization 5-20%, ASR 20-55%, rest follows

## Phase 1: Adaptive Pipeline Profiles

### Config changes (`backend/app/config.py`):
- Add `PipelineProfile` enum: SMALL, LARGE, AUTO (default: AUTO)
- Add to ASRConfig: `initial_prompt: str = ""`, `condition_on_previous_text: bool = False`
- Add to DiarizationConfig: `window_duration: float = 1.0`, `step_duration: float = 0.5` (currently hardcoded in diarization.py line 671)
- Make summarization `top_p` and `repetition_penalty` configurable (currently hardcoded)
- Add to upload API: optional `expected_speakers: int` parameter
- Add to frontend upload page: "Expected number of speakers" dropdown (optional)

### Profile parameter values:

**SMALL profile (2-4 speakers)** — keep current defaults:
- ASR: beam_size=5, vad_threshold=0.5, vad_min_speech_ms=250, condition_on_previous_text=False
- Diarization: window=1.0s, step=0.5s
- Summarization: standard single-pass prompts, chunk_overlap=1500

**LARGE profile (5+ speakers)** — tuned for complex meetings:
- ASR: beam_size=7, vad_threshold=0.35, vad_min_speech_ms=150, vad_min_silence_ms=80, condition_on_previous_text=False (NOT True — with many speakers, previous text context leaks between speakers and causes cross-contamination)
- Diarization: window=0.75s, step=0.25s
- Summarization: chunk_overlap=2500, QA-prompting strategy (see Phase 4), repetition_penalty=1.15

### Diarization always runs with aggressive settings:
Over-segmentation (too many speakers) is easy to fix — merge similar speakers.
Under-segmentation (missing speakers) is IMPOSSIBLE to fix.
So diarization should always lean toward fine-grained detection.

If user provides `expected_speakers`, pass it to constrain min/max speaker range.

## Phase 2: Bugs to Fix (quality-critical)

These bugs directly affect output quality. Fix them before tuning parameters.

### Bug 1: Summarization truncates 33% of input silently
**File:** `summarization.py` line ~486
**Problem:** `max_length=8000` in tokenizer, but `chunk_size=12000` in config. Each chunk is ~12K tokens but the tokenizer truncates to 8K before generation — silently dropping ~33% of the transcript chunk.
**Fix:** Change `max_length` to use `config.summarization.chunk_size` or a new `max_input_tokens` config value. Ensure it matches the model's actual context window.

### Bug 2: Silhouette validation uses wrong clustering algorithm
**File:** `diarization.py` lines ~602-624
**Problem:** Speaker count estimation uses AgglomerativeClustering for silhouette scoring, but actual clustering uses SpectralClustering. The "best k" found by one algorithm may be wrong for the other.
**Fix:** Use the same clustering method for both validation and final clustering. If method is "spectral", validate with spectral. If "agglomerative", validate with agglomerative.

### Bug 3: Chunk overlap calculated by lines, not tokens
**File:** `summarization.py` lines ~720-745
**Problem:** Chunking logic counts tokens but overlap calculation works on line count (`current_chunk[-overlap_size:]` where overlap_size is line count). A single long line could be hundreds of tokens.
**Fix:** Calculate overlap in tokens, consistent with chunking.

### Bug 4: CPU compute type invalid
**File:** `whisper_asr.py` line ~78
**Problem:** `"int8"` is not a valid CTranslate2 standalone compute type for CPU. Valid: `float32`, `int8_float32`, `int8_float16`.
**Fix:** Change to `"int8_float32"` for CPU or use config value instead of hardcoding.

### Bug 5: Aligner falls back to first speaker instead of closest
**File:** `aligner.py` lines ~93-98
**Problem:** If no time overlap found between transcription segment and any diarization segment, falls back to `diarization[0].speaker_id` (first speaker). Should find the CLOSEST diarization segment by time proximity.
**Fix:** Find nearest diarization segment by time distance instead of defaulting to first.

## Phase 3: Transcription Improvements

In `whisper_asr.py`, update the `transcribe()` call:

1. **Add `initial_prompt`** — configurable via `config.asr.initial_prompt`. Default:
   `"Протокол совещания юридической фирмы EPAM. Участники обсуждают правовые вопросы, контракты, судебные дела."`

2. **Make whisper model configurable via `config.asr.whisper_model`** — The `_get_whisper_model_size()` currently only maps standard size names. If whisper_model is a full HuggingFace path (like `bzikst/faster-whisper-large-v3-russian`), pass it through directly instead of trying to extract a size keyword. This allows switching to Russian-finetuned Whisper with just a config change.

3. **Use config.asr.compute_type instead of hardcoding** — currently line 78 ignores the config value.

4. **Add `no_speech_threshold` and `compression_ratio_threshold`** to ASRConfig with Whisper defaults. Pass to transcribe().

## Phase 4: Diarization Improvements

**Option A (preferred): Integrate pyannote-audio**
- `pip install pyannote-audio` (add to requirements.txt)
- Create `backend/engine/pyannote_diarization.py` — new engine implementing BaseEngine
- Use `pyannote/speaker-diarization-3.1` or `community-1`
- Config: `diarization.engine: pyannote|speechbrain` (default: pyannote)
- ~600MB VRAM, fits in sequential loading budget
- Keep existing SpeechBrain engine as fallback
- pyannote natively supports `min_speakers` and `max_speakers`

**Option B (if pyannote has issues): Tune existing pipeline**
- Move window/step from hardcoded to config-driven (currently lines 671-673)
- Fix the silhouette/clustering algorithm mismatch (Bug 2 above)
- Lower silhouette acceptance — current logic picks k with HIGHEST silhouette, biasing toward fewer clusters
- Add minimum acceptable silhouette threshold — if score at k=11 is 0.3 and k=7 is 0.35, prefer 11 (more speakers is safer for meetings)

## Phase 5: Summarization Improvements

1. **Try Qwen3-8B** — change model config to `Qwen/Qwen3-8B`. Test with 4-bit. Same API. Revert if broken.

2. **Add `repetition_penalty=1.15`** to generation parameters. Russian protocol text is highly prone to repetitive phrasing. This is currently not set at all.

3. **Make `top_p` configurable** (currently hardcoded 0.9 in summarization.py line 496). Move to config.

4. **Add few-shot example** — add ONE short example of a good protocol section to the Russian system prompt.

5. **QA-prompting for LARGE profile** — instead of one "summarize" prompt, send 4 focused prompts:
   - "Перечисли ВСЕХ участников и кратко опиши позицию каждого."
   - "Какие решения были приняты? Кто ответственный?"
   - "Какие задачи поставлены? Кому, с каким сроком?"
   - "Какие вопросы остались нерешёнными?"
   Combine answers into 7-section protocol. Research shows ~28% quality improvement.

6. **Anti-hallucination instruction** — add to ALL prompts:
   `"СТРОГО ЗАПРЕЩЕНО добавлять информацию, которой нет в стенограмме. Если информация неясна, пиши 'не установлено'."`

## Testing

- Before starting: backup to `VERITAS_backups/backup_block5d_YYYY-MM-DD/`
- After EACH phase, test on the same 11-speaker recording
- Save results in `data/test_results/block5d/` with timestamps
- Compare: detected speaker count, transcript coherence, protocol quality
- Log profile selection in audit trail

## Implementation Order
1. Pipeline reorder (diarization before ASR) + Silero VAD integration + profile infrastructure
2. Bug fixes (Phase 2 — all 5 bugs)
3. Transcription tuning (Phase 3)
4. Diarization upgrade (Phase 4)
5. Summarization upgrade (Phase 5)
Commit after EACH working step. Test after each step.

## Rules
- Read CLAUDE.md Session Instructions before starting
- All code in English, no Cyrillic in source (Cyrillic only in string literals for prompts/filler lists)
- Update CLAUDE.md before session ends with ALL changes made
- Production-ready code — no stubs, no placeholders
- Security: no new external network calls, all models from local cache
