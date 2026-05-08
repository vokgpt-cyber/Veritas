# EPAM VERITAS — Development State

**Verbal Intelligence, Transcription and Summarization**
On-premise meeting transcription and protocol generation for EPAM law firm.

## Current State: MVP Demo Ready (offline test passed 2026-04-08)

## Architecture

- **Pipeline Order** (MVP): Preprocess -> ASR -> Diarization -> Align -> Post-process -> QA -> [Summarize] -> Format
  - This is the proven pre-5d order. Block 5d tried VAD pre-filtering and adaptive profiles but degraded quality — reverted.
  - Summarize stage is skippable via `summarization.enabled: false` (transcript-only mode)
  - Dead Block 5d code removed: VAD instantiation, `_select_profile()`, old structured protocol methods
- **ASR**: Dual-engine, auto-detected by platform:
  - **Windows**: faster-whisper (`large-v3` or configurable HF repo, CTranslate2) — language pinned to `ru`
  - **Linux/Docker**: NVIDIA NeMo Conformer (`stt_ru_conformer_transducer_large`)
  - Config: `engine: auto|nemo|whisper` in ASRConfig
  - Whisper model configurable via `whisper_model` — supports HF repo paths
  - ASR uses only proven parameters: `language`, `beam_size`, `word_timestamps`, `vad_filter`, `vad_parameters`
  - Block 5d params (`initial_prompt`, `condition_on_previous_text`, `no_speech_threshold`, `compression_ratio_threshold`) were REVERTED — they degraded punctuation and quotation marks
- **Audio Preprocessing**: Dual backend — ffmpeg (primary) with torchaudio fallback for old/missing ffmpeg
- **Diarization**: Dual-engine:
  - **SpeechBrain** (default): ECAPA-TDNN + spectral/agglomerative clustering, silhouette scoring with same-algorithm validation, config-driven window/step
  - **pyannote-audio** (optional): `pyannote/speaker-diarization-3.1`, state-of-the-art DER 11-19%, ~600MB VRAM, native min/max_speakers support
  - Config: `diarization.engine: speechbrain|pyannote`
- **Post-processing**: Rule-based filler removal (RU/EN), sentence cleanup, language detection (no VRAM needed)
- **Summarization**: Qwen3-8B via llama-cpp-python (GGUF Q4_K_M, ~4.5GB weights + ~0.8GB KV cache)
  - Migrated from transformers+bitsandbytes (Qwen3-4B) to llama-cpp-python (Qwen3-8B) for 2x model capacity in same VRAM
  - GGUF format: single file in `models/` directory, efficient KV cache, no bitsandbytes dependency
  - Context window: 4096 tokens (chunk_size=3000 input + max_new_tokens=800 output)
  - MVP approach: ONE paragraph summary (3-5 sentences), NOT structured 7-section protocol
  - Multi-chunk transcripts: each chunk summarized, then consolidated into single paragraph
  - Anti-hallucination instructions in all prompts
  - `summarization.enabled: false` skips LLM entirely for transcript-only DOCX output
  - Old dead code removed: `_build_messages`, `_build_qa_messages_ru`, `_parse_protocol`
- **Backend**: Python 3.11, FastAPI, WebSocket for progress + HTTP polling fallback
- **Auth**: JWT via python-jose + bcrypt password hashing
- **Frontend**: React 19 + Vite 8 + TailwindCSS 4 + lucide-react icons
- **Hardware**: RTX 4060 (8 GB VRAM), sequential model loading
- **Docker**: NVIDIA CUDA 12.1 base image, non-root user, hardened
- **Security**: AES-256-GCM encryption at rest, structured audit logging, auto cleanup

## Key Constraints

- **Privacy/Security**: CRITICAL. Law firm data. All processing on-premise, no data leaves the machine. No public code. No untrusted dependencies. Authentication required.
- **VRAM**: 8 GB limit — models load/unload sequentially via VRAMManager
- **Speakers**: Practical MVP limits ~30 min audio, ~15 speakers (no hard limits enforced)
- **Language**: All code in English, no Cyrillic in source files. Docstrings in English.
- **Branding**: EPAM red #B2001F, Georgia headings, Arial Narrow body

## Backend Structure (~12,000 lines, 43 files)

```
backend/
  app/
    auth.py          — JWT authentication, token creation/validation, user management
    config.py        — Pydantic Settings with YAML support, env var overrides
    main.py          — FastAPI lifespan, CORS, route registration
    models.py        — Pydantic models for jobs, segments, protocols
    websocket.py     — WebSocket progress reporting
    routes/
      auth.py        — Login and token verification endpoints
      meetings.py    — Upload, process, status, download endpoints (JWT protected)
      speakers.py    — Speaker management endpoints (JWT protected)
      system.py      — Health check, GPU status, config endpoints (JWT protected)
  core/
    orchestrator.py  — Main pipeline coordinator, manages full processing flow
    vram_manager.py  — GPU VRAM allocation, sequential model load/unload
    audio.py         — Audio preprocessing (ffmpeg primary, torchaudio fallback)
    aligner.py       — Aligns ASR output with diarization segments
    formatter.py     — DOCX/JSON/MD protocol generation with EPAM branding
    qa.py            — Quality assurance validation for each pipeline stage
    encryption.py    — AES-256-GCM encryption at rest for audio/transcripts
    audit.py         — Structured JSON audit logging for security events
    cleanup.py       — Automatic temp file cleanup after processing
    postprocessor.py — Transcript post-processing: filler removal, sentence cleanup, language detection
  engine/
    base.py          — Abstract base class for all ML engines
    vad.py           — Silero VAD engine (now active in pipeline, runs before diarization)
    asr.py           — NeMo Conformer ASR engine (Linux)
    whisper_asr.py   — faster-whisper ASR engine (Windows, default), configurable model/params
    diarization.py   — SpeechBrain ECAPA-TDNN diarization engine (config-driven window/step)
    pyannote_diarization.py — pyannote-audio diarization engine (optional, state-of-the-art)
    summarization.py — Qwen3-4B summarization engine (QA-prompting, anti-hallucination)
    mock.py          — Mock engines for testing without GPU
  tests/
    conftest.py      — Test configuration and shared fixtures
    test_aligner.py  — Transcript alignment tests (6 tests)
    test_api.py      — API endpoint tests with JWT auth (27 tests)
    test_auth.py     — JWT authentication tests (11 tests)
    test_config.py   — Configuration tests (12 tests)
    test_engines.py  — ML engine tests (14 tests)
    test_formatter.py — DOCX/JSON formatter tests (5 tests)
    test_models.py   — Pydantic model tests (20 tests)
    test_qa.py       — Quality assurance tests (12 tests)
    test_encryption.py — AES-256-GCM encryption tests (19 tests)
    test_audit.py    — Audit logging tests (11 tests)
    test_cleanup.py  — File cleanup tests (9 tests)
    test_pipeline_e2e.py — E2E pipeline, QA validation, alignment tests (27 tests)
    test_performance.py  — Benchmarks, VRAM manager, audio preprocessing (32 tests)
    test_edge_cases.py   — Edge cases, error recovery, boundary conditions (30 tests)
    test_docker.py       — Docker/compose security, config validation (21 tests)
```

## Config

- `config/settings.yaml` — Full configuration with all parameters documented
- Environment variables with `EPAM_` prefix override YAML values
- Auth config: `EPAM_AUTH_SECRET_KEY`, `EPAM_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES`
- Encryption config: `EPAM_ENCRYPTION_SECRET_KEY`, `EPAM_ENCRYPTION_ENABLED`
- Audit config: `EPAM_AUDIT_ENABLED`, `EPAM_AUDIT_LOG_DIR`
- Cleanup config: `EPAM_CLEANUP_AUTO_CLEANUP`, `EPAM_CLEANUP_RETENTION_HOURS`
- Security config: `EPAM_SECURITY_CORS_PRODUCTION_ORIGINS`
- Pipeline config (Block 5d):
  - `EPAM_PIPELINE_PROFILE`: auto|small|large (default: auto)
  - `EPAM_PIPELINE_LARGE_PROFILE_THRESHOLD`: speaker count threshold for LARGE (default: 5)
  - `EPAM_ASR_INITIAL_PROMPT`: domain vocabulary hint for Whisper
  - `EPAM_ASR_WHISPER_MODEL`: model name or HF repo path
  - `EPAM_ASR_CONDITION_ON_PREVIOUS_TEXT`: use previous chunk context (default: false)
  - `EPAM_ASR_NO_SPEECH_THRESHOLD`, `EPAM_ASR_COMPRESSION_RATIO_THRESHOLD`: Whisper tuning
  - `EPAM_DIARIZATION_ENGINE`: speechbrain|pyannote (default: speechbrain)
  - `EPAM_DIARIZATION_WINDOW_DURATION`, `EPAM_DIARIZATION_STEP_DURATION`: sliding window params
  - `EPAM_SUMMARIZATION_ENABLED`: true/false — set false for transcript-only output (no LLM)
  - `EPAM_SUMMARIZATION_GGUF_MODEL_PATH`: explicit path to .gguf file (empty = auto-detect in models/)
  - `EPAM_SUMMARIZATION_CONTEXT_SIZE`: context window (default: 4096, keep <=4096 for 8GB VRAM)
  - `EPAM_SUMMARIZATION_TOP_P`, `EPAM_SUMMARIZATION_REPETITION_PENALTY`: generation tuning

## Authentication

- JWT tokens via python-jose (HS256 algorithm)
- bcrypt password hashing via passlib
- Default admin user (admin/admin) — MUST be changed in production
- Token expiry: 8 hours (configurable via env)
- Public endpoints: /api/health, /api/auth/login, /api/auth/verify, /api/docs
- All other endpoints require valid JWT Bearer token
- GPU release requires admin role

## Dependencies (all from trusted sources)

- fastapi, uvicorn, pydantic — Web framework (Sebastian Ramirez / Pydantic team)
- faster-whisper — CTranslate2 Whisper (ASR on Windows, Guillaume Klein / SYSTRAN)
- nemo_toolkit — NVIDIA (ASR on Linux, optional on Windows)
- speechbrain — Mila / Cambridge (diarization, SpeechBrain engine)
- pyannote-audio — CNRS/IRIT (optional, pyannote diarization engine, state-of-the-art)
- llama-cpp-python — GGUF LLM inference via llama.cpp (Qwen3-8B Q4_K_M)
- torch, torchaudio — Meta/PyTorch Foundation (torchaudio also used as ffmpeg fallback for audio preprocessing)
- librosa, soundfile — Audio processing (standard scientific Python)
- python-docx — Document generation (well-maintained, MIT license)
- python-jose[cryptography] — JWT tokens (BSD license)
- passlib[bcrypt] — Password hashing (BSD license)
- scikit-learn — Clustering for diarization (standard ML library)
- nvidia-ml-py3 — GPU monitoring (NVIDIA official)
- cryptography — AES-256-GCM encryption at rest (PSF/cryptography.io)

## Test Coverage (241 tests passing, 7 skipped)

### Blocks 1-4 Tests (131 tests)
- Config: YAML loading, env overrides, defaults, branding (12 tests)
- Models: all 15 Pydantic models, serialization, validation (20 tests)
- Engines: VAD, ASR, diarization, summarization (14 tests, require torch)
- Aligner: time-overlap alignment, chronological ordering, edge cases (6 tests)
- Formatter: DOCX generation with EPAM branding, JSON roundtrip (5 tests)
- QA: audio/transcription/diarization/alignment/protocol validation (12 tests)
- API: public endpoints, auth flow, protected endpoints, token validation (27 tests)
- Auth: password hashing, token creation/decode, user authentication (11 tests)
- Encryption: AES-256-GCM file/bytes roundtrip, wrong key, corruption (19 tests)
- Audit: event creation, JSON logging, rotation, disabled mode (11 tests)
- Cleanup: intermediate removal, orphan cleanup, retention, secure delete (9 tests)

### Block 5 Tests (110 tests, 7 skipped)
- E2E pipeline: full flow with mock engines, progress tracking, output files (5 tests)
- Pipeline stages: alignment, merge consecutive, participants, DOCX/JSON (6 tests)
- QA validation: threshold tuning, hallucination detection at scale (12 tests)
- Alignment validation: order, time range, empty input (4 tests)
- Performance: alignment scaling (50-1000 segments), merge speed, QA speed, DOCX/JSON throughput (14 tests)
- VRAM manager: singleton, status, registration, availability, release (7 tests)
- Concurrent: sequential multi-job processing (1 test)
- Audio preprocessing: metadata, conversion, validation (5 tests, 4 require ffprobe)
- Edge cases: short audio, silence-heavy, 1 speaker, 20 speakers, boundary conditions (21 tests)
- Error recovery: retry on failure, permanent failure, VRAM unavailable (4 tests)
- Intermediate results: save/load roundtrip (2 tests)
- Docker/config: Dockerfile security (6), compose security (8), config validity (7) (21 tests)

## Development Blocks

### Block 1: Backend Core — DONE
Config, models, all 5 ML engines, orchestrator, API, Docker, start scripts.

### Block 2: Integration Testing + Basic Security — DONE
- Fixed all import errors across 19 modules
- Added `from __future__ import annotations` for robust type hints
- Fixed orchestrator config passing bug (sub-config vs full config)
- Fixed DOCX formatter bug (get_or_add_pPr -> get_or_add_tcPr)
- Fixed spectral clustering NaN with embedding normalization
- Migrated deprecated Pydantic v1 @validator to v2 @field_validator
- Added JWT authentication with python-jose + passlib
- Protected all API routes except health/auth/docs
- Added admin-only role guard for GPU release
- 107 integration tests across 8 test modules
- Synthetic test audio: `data/test/test_meeting_ru.wav` (10s, 16kHz, mono)
- .gitignore covers all sensitive data (audio, transcripts, env, models)

### Block 3: React Frontend — DONE
- Login screen with JWT auth + graceful fallback when auth not enforced
- Upload screen with drag-and-drop, format/size validation
- Progress stepper with WebSocket real-time updates + fallback polling
- Interactive transcript editor: inline text editing, speaker reassignment
- Protocol viewer with DOCX/JSON download
- Meetings list with status badges, progress bars, delete
- Speakers management page
- System status dashboard: CPU, RAM, GPU/VRAM, temperature, job stats
- EPAM branding: #B2001F red, Georgia headings, Arial Narrow body
- Responsive sidebar layout with mobile menu
- Vite dev proxy for /api and /ws to backend :8000

#### Frontend Structure (14 source files)
```
frontend/
  src/
    api/client.ts          — HTTP client with JWT auth, all API endpoints
    context/AuthContext.tsx — Auth state, login/logout/skipAuth
    hooks/useWebSocket.ts  — WebSocket hook for real-time progress
    types/api.ts           — TypeScript types matching backend models
    components/
      Layout.tsx           — Sidebar navigation + mobile menu
      Logo.tsx             — VERITAS by EPAM logo
    pages/
      LoginPage.tsx        — JWT login with backend-not-ready fallback
      UploadPage.tsx       — Drag-and-drop audio upload
      ProcessingPage.tsx   — Pipeline stepper with WebSocket progress
      TranscriptPage.tsx   — Interactive transcript editor
      ProtocolPage.tsx     — Protocol viewer + DOCX/JSON download
      MeetingsPage.tsx     — Meetings list with status/actions
      SpeakersPage.tsx     — Speaker profile management
      SystemPage.tsx       — System health + GPU monitoring
    App.tsx                — Router with auth guards
    main.tsx               — Entry point
    index.css              — Tailwind 4 + EPAM custom theme
```

### Block 4: Security Hardening & Dependency Audit — DONE
- AES-256-GCM encryption at rest for audio and transcripts
  - PBKDF2-SHA256 key derivation (600K iterations, OWASP 2023)
  - Per-file random salt + nonce, authenticated encryption
  - Secure file deletion (overwrite with random data before unlink)
  - Configurable via EPAM_ENCRYPTION_SECRET_KEY env var
- Structured JSON audit logging (JSONL format, SIEM-ready)
  - Auth events (login success/failure), meeting ops (upload/download/delete)
  - System events (startup/shutdown), security events (unauthorized access)
  - Date-based rotation, size limits, configurable retention
- Automatic temp file cleanup after processing
  - Removes intermediate files (raw audio, transcription, diarization)
  - Preserves final outputs (DOCX, JSON protocol, edited transcript)
  - Orphan detection for abandoned jobs, configurable retention
  - Secure delete with random overwrite before unlink
- Dependency pinning with exact versions (all packages pinned)
- Docker hardening
  - Non-root user (veritas), all capabilities dropped except SYS_NICE
  - Read-only root filesystem, size-limited tmpfs, no host /tmp mount
  - curl/wget/git removed from final image
  - TLS certificate mount point (/app/certs), resource limits
  - Port bound to localhost only (use reverse proxy for external)
  - no-new-privileges security option
- CORS lockdown for production
  - Configurable production origins via env var
  - Restricted methods (GET/POST/PUT/DELETE/OPTIONS only)
  - Restricted headers (Authorization, Content-Type, X-Requested-With)
- Network isolation: Docker bridge network, compose-level isolation
- SecurityConfig added to AppConfig with full env var support
- 39 new tests (19 encryption + 11 audit + 9 cleanup), all passing

### Block 5: End-to-End Testing, Optimization & QA — DONE
- Full E2E pipeline test with mock engines (real ML engines require GPU)
  - Complete pipeline flow: preprocessing -> ASR -> diarization -> alignment -> summarization -> formatting
  - Stage transitions verified: all 8 stages + progress reporting
  - VRAM lifecycle: check -> load -> process -> unload -> release for each engine
  - Intermediate result saving and loading for recovery
  - Job CRUD: create, list, get, delete
- Performance benchmarking (all within limits):
  - Alignment: O(N*M) confirmed fast up to 1000 segments (<2s)
  - Merge consecutive: O(N) confirmed (<0.5s for 500 segments)
  - DOCX generation: <5s for 500-segment protocols
  - JSON generation: <1s for 500 segments
  - QA validation: <1s for 1000 transcription segments
  - Hallucination check: <2s on 200 normal segments
- VRAM optimization verified:
  - Singleton pattern ensures single VRAMManager instance
  - Model registration/unregistration lifecycle tested
  - Graceful fallback when no GPU (dummy status with 24GB)
  - wait_for_available returns immediately when VRAM sufficient
  - release_all works without torch/CUDA
- Edge case testing:
  - Very short audio (1s, 0s) — correctly rejected by QA
  - Silence-heavy recordings — empty transcription/diarization handled
  - Single speaker — passes with warning, 100% speaking share
  - 20 speakers — alignment, participant stats, DOCX all work correctly
  - Zero-duration segments, overlapping segments — no crashes
  - Very long text (50K chars), 1000 micro-segments — handled
  - Unicode/Russian text — JSON roundtrip and DOCX generation work
- Error recovery:
  - Retry on preprocessing failure — succeeds after retry
  - Retry on ASR failure (OOM simulation) — succeeds after retry
  - Permanent failure after max retries — error state with message
  - VRAM unavailable — fails with clear VRAM error
- Docker verification:
  - Dockerfile: CUDA base, non-root user, no curl/wget, healthcheck
  - docker-compose: security_opt, resource limits, read-only rootfs, dropped caps, tmpfs, network isolation
  - Config: settings.yaml valid and complete, .env.example with critical vars, requirements pinned
  - .gitignore covers sensitive files (.env, audio, models)
- Known issues found and documented:
  - Hallucination detection loop is empty with default max_repeat_length=50 (range(50,50) = empty)
    - Fix: increase max_repeat_length to 100+ in settings.yaml, or lower loop start from 50 to 20
  - ProtocolFormatter uses @classmethod pattern (no __init__ args), inconsistent with other components
  - Audio preprocessing requires modern ffprobe with -print_json support (torchaudio fallback now handles this)
  - 14 engine tests require torch/CUDA (cannot run in CI without GPU)

### First Real Pipeline Run (29-min Russian legal discussion) — DONE
Successfully ran full pipeline on Windows with real audio (YouTube "Адвокатский клуб", 29 min, 2 speakers).
Results: 437 ASR segments, avg confidence 0.95+, 2 speakers correctly identified, Russian protocol generated.

**Runtime bugs found and fixed:**
1. **ffprobe `-print_json` not supported on old Windows ffprobe**: Added `_ffmpeg_available()` that tests the flag, plus runtime fallback to torchaudio in `audio.py`
2. **torchaudio fallback**: Added `_get_metadata_torchaudio()` and `_convert_torchaudio()` methods for resampling + mono conversion when ffmpeg unavailable
3. **SpeechBrain symlink error `[WinError 1314]`**: Windows requires Developer Mode for symlinks. Added `_patch_speechbrain_symlinks()` in `diarization.py` that monkey-patches `link_with_strategy` to copy files instead. First version missed `return destination` → fixed.
4. **Progress callback signature mismatch**: Engine callbacks pass `(progress, message)` but orchestrator callbacks only accepted `(progress)`. Added `message: str = ""` to all three progress callbacks in `orchestrator.py`.
5. **`TranscriptionSegment` not subscriptable**: Diarization expected dicts but got Pydantic objects. Added conversion in orchestrator before passing to diarization.
6. **Summarization returned prompt template text**: `_generate()` decoded full sequence including prompt. Fixed to decode only `outputs[0][input_length:]` (new tokens only).
7. **Progress math bug**: Callbacks receive 0.0-1.0 float, but multipliers were `* 0.25` instead of `* 25.0`. Fixed all three stage multipliers in orchestrator.
8. **WebSocket progress not pushing**: Background task runs in separate event loop, can't push WebSocket from there. Changed to hybrid: WebSocket sends `status` requests every 5s + HTTP polls every 8s as fallback.
9. **Endpoints fail after backend restart**: In-memory job state lost. Fixed transcript and protocol endpoints to check files on disk as fallback when job not in memory.
10. **Prompt text leaking into protocol tasks**: Summarization output parser now filters prompt header artifacts (`skip_phrases`).

### Block 5c: Polish & Bug Fixes — DONE
**Offline-first model loading (critical for on-premise):**
- All 5 ML engines now work fully offline after initial model download
- SpeechBrain: loads from local cache with patched hyperparams.yaml, empty custom.py
- Summarization (Qwen): `local_files_only=True` passed to `from_pretrained()`, eliminates HF retry noise
- faster-whisper: `HF_HUB_OFFLINE=1` when cache detected
- NeMo ASR: defaults to offline, falls back to network only on first run
- Silero VAD: `source="local"` from torch hub cache when available

**Protocol download fix:**
- Download buttons now use fetch + blob instead of direct `<a href>` links
- Fixes "Файл недоступен" error caused by missing JWT token in browser navigation

**Diarization accuracy improvement:**
- Replaced eigenvalue gap heuristic with silhouette score validation
- Tries multiple cluster counts (min_speakers to max_speakers), picks highest silhouette score
- Window size reduced from 1.5s to 1.0s (step 0.5s) for finer speaker granularity
- Candidate range widened: max_k = embeddings // 5 (was // 10) to detect more speakers

**Transcript post-processing (new `postprocessor.py`):**
- Language detection: Cyrillic vs Latin character counting
- Russian filler removal: 25+ filler words/phrases (ну, вот, значит, как бы, типа, etc.)
- English filler removal: 14 filler words (uh, um, like, you know, etc.)
- Whisper hallucination cleanup: removes repeated word/phrase artifacts
- Sentence capitalization and whitespace normalization
- Short/empty segment merging with adjacent same-speaker segments
- Integrated into orchestrator between alignment (Stage 4) and QA (Stage 5)

**Russian protocol template:**
- Summarization prompts fully rewritten in Russian for Russian-language meetings
- Section headers in Russian: ТЕМА СОВЕЩАНИЯ, УЧАСТНИКИ, КРАТКОЕ СОДЕРЖАНИЕ, etc.
- DOCX formatter uses Russian labels when content is Russian (auto-detected)
- Date format switched to DD.MM.YYYY for Russian protocols
- Detailed, task-specific prompts for Qwen2.5-7B (law firm context, no hallucination)
- Parser handles both RU and EN section headers for robustness

**Speaker renaming:**
- New API endpoint: PUT /api/meetings/{job_id}/speakers/rename?old_name=X&new_name=Y
- Renames speaker across all transcript segments in aligned.json
- Also updates protocol.json (transcript + participants sections)
- Enables replacing "Speaker_1" with real names like "Ivanov A.V."

**Results archiving:**
- Orchestrator copies final outputs to `data/archive/YYYY-MM-DD_filename/` after processing
- Archive contains: DOCX protocol, JSON protocol, transcript.json
- Accessible without knowing internal job UUIDs

**Second real-world E2E test (on-premise, no internet, 2026-04-07):**
- Pipeline ran end-to-end on Windows with RTX 4060, all models loaded from local cache
- ALL FOUR areas had quality issues:
  - **Diarization**: 11 real speakers → detected only 7 (under-segmentation, merging distinct speakers)
  - **Transcription**: Quality issues with Whisper large-v3 on Russian legal speech
  - **Summarization**: Qwen2.5-7B protocol output quality was poor
  - **Frontend**: UI bugs during the flow (details TBD — needs focused QA pass)
- These are quality tuning issues, not architecture failures — the pipeline works end-to-end

### Block 5d: ML Quality Upgrade — DONE (then partially REVERTED in MVP Stabilization)

**What was built in Block 5d (code exists but some features parked):**
- PipelineProfile enum (SMALL, LARGE, AUTO) and PipelineConfig added to AppConfig
- `_select_profile()` method in orchestrator — NOT called in MVP pipeline (parked)
- Silero VAD instantiated in orchestrator — NOT called in MVP pipeline (parked)
- `pyannote_diarization.py` created — optional engine, works but not default
- `initial_prompt`, `condition_on_previous_text`, `no_speech_threshold`, `compression_ratio_threshold` added to ASRConfig — NOT passed to Whisper in MVP (reverted, degraded quality)
- QA-prompting mode (`_build_qa_messages_ru`) — parked dead code, too ambitious for Qwen3-4B
- Old structured protocol methods (`_build_messages`, `_build_messages_ru`, `_build_messages_en`) — parked dead code

**Bug fixes from Block 5d (these ARE active and important):**
- Bug 25: Summarization tokenizer truncated 33% of input silently. Fixed to use config chunk_size.
- Bug 26: Silhouette validation used wrong clustering algorithm. Fixed to use same for both.
- Bug 27: Chunk overlap calculated by lines not tokens. Fixed.
- Bug 28: CPU compute type `"int8"` invalid for CTranslate2. Fixed to `"int8_float32"`.
- Bug 29: Aligner fell back to first speaker instead of closest. Fixed.

**Other Block 5d improvements still active:**
- `whisper_model` configurable via HF repo paths
- Config `compute_type` actually used now (was hardcoded)
- SpeechBrain window/step config-driven (was hardcoded)
- Silhouette validation uses same clustering algorithm as final
- Qwen3-4B deployed (Qwen3-8B exceeded 8GB VRAM)
- `expected_speakers` parameter in upload API and frontend UI

### MVP Stabilization (2026-04-08) — DONE

**Reverted Block 5d changes that degraded quality:**
- Pipeline order reverted to pre-5d: Preprocess -> ASR -> Diarization -> Align -> Post-process -> QA -> Summarize -> Format
- Whisper ASR reverted to minimal parameters (language, beam_size, word_timestamps, vad_filter, vad_parameters only)
- VAD pre-filtering stage removed from pipeline (Silero still instantiated but not called)
- Adaptive profile selection removed from pipeline (_select_profile not called)

**Summarization simplified to 1-paragraph MVP (then migrated to llama-cpp-python):**
- Originally: Qwen3-4B via transformers+bitsandbytes, 1-paragraph summary
- Migrated to: Qwen3-8B via llama-cpp-python (GGUF Q4_K_M), 2x model capacity, same VRAM
- Methods: `_build_summary_prompt()`, `_build_consolidation_prompt()`, `_extract_topic()`
- Multi-chunk: each chunk gets brief summary, then consolidated into single paragraph
- Context window: 4096 (chunk_size=3000 + max_new_tokens=800)
- `summarization.enabled: false` config flag for transcript-only mode (no LLM)

**Formatter simplified:**
- Always Russian headers and DD.MM.YYYY dates
- Protocol: title page + participants table + summary paragraph + transcript
- Structured sections shown ONLY if LLM produced content (empty by default in MVP)

**Bug fixes (Bugs 30-41):**
- Bug 30: CTranslate2 CUDA deadlock on Whisper unload — parking pattern
- Bug 31: VRAM leak — gc/empty_cache in wrong order across all 4 engines
- Bug 32: VRAMManager release_all() deadlock — removed torch.cuda.synchronize()
- Bug 33: transformers 5.x incompatible with bitsandbytes — pinned transformers==4.57.0
- Bug 34: Qwen3-8B OOM — switched to Qwen3-4B (~2.5GB 4-bit)
- Bug 35: device_map="auto" unnecessary for 4B model — switched to {"": 0}
- Bug 36: Config YAML never loaded — fixed path to config/settings.yaml
- Bug 37: settings.yaml initial_prompt was Latin transliteration — removed from YAML
- Bug 38: Summarization OOM — chunk_size=3500, max_new_tokens=1500
- Bug 39: Consolidation truncated — _generate() accepts max_input_tokens/max_new_tokens overrides
- Bug 40: Orchestrator retries useless after OOM — gc.collect() + empty_cache() between retries
- Bug 41: DOCX key topics duplicated — show content only

**Job queue guard added:**
- Upload endpoint rejects new jobs while another is processing (HTTP 409)
- Single-GPU hardware can only process one meeting at a time

**Dead code cleanup:**
- Removed `_select_profile()` from orchestrator (was Block 5d, never called)
- Removed VAD instantiation from orchestrator (Silero VAD no longer loaded)
- Removed `PipelineProfile` import from orchestrator
- Removed `_build_messages`, `_build_qa_messages_ru`, `_build_messages_ru`, `_build_messages_en`, `_parse_protocol` from summarization.py
- Removed unused imports: `Decision`, `TaskItem`, `TopicItem` from summarization.py

**Bug fix: llama-cpp-python Windows CUDA DLL loading (Bug 42):**
- llama-cpp-python's ggml-cuda.dll needs cublas64_12.dll and cudart64_12.dll at load time
- No CUDA toolkit installed on laptop — but PyTorch bundles these DLLs in torch/lib/
- Fixed: `_setup_cuda_dll_paths()` in summarization.py adds PyTorch lib dir via `os.add_dll_directory()` before importing llama_cpp
- Windows prebuilt wheel: `dougeeai/llama-cpp-python-wheels` v0.3.16 (cu12.1, sm75, py311)

**Known remaining issues:**
- Diarization under-segments speakers (11 real -> 7 detected)
- ProtocolPage shows transcript but summary stats show "0 speakers, 0 segments"
- MeetingsPage only shows in-memory jobs, not completed jobs from disk
## Default Capabilities (PLANNED — not yet implemented)

These features are designed into the architecture but not yet built:
- **Multilingual**: Russian primary, English auto-detected. Mixed-language meetings planned.
- **Smart Protocol Templates**: Auto-detect meeting type (client, internal, board) — PLANNED for future blocks.
- **Recap**: One-click brief from previous meetings with same participants/case — PLANNED for Block 8 (Legal Intelligence).

## MVP Demo Optimization — IN PROGRESS

Goal: make the MVP demo on ASUS ROG Zephyrus G14 (RTX 4060 8GB) as impressive as possible.
Sweet spot: 2-4 speakers, ~10-30 min audio, Russian. This is where the system shines.

**Done (Phase 1 — infrastructure):**
- Summarization migrated to llama-cpp-python + Qwen3-8B GGUF Q4_K_M (2x model capacity)
- `summarization.enabled` toggle for transcript-only fallback
- Dead Block 5d code cleaned up (VAD, profiles, old protocol methods)
- Job queue guard: rejects uploads while another job processes (HTTP 409)
- llama-cpp-python installed with CUDA (v0.3.16 prebuilt wheel, dougeeai)
- Qwen3-8B GGUF downloaded and tested on GPU (load + generate + unload OK)
- Bug 42: Windows CUDA DLL loading — PyTorch lib dir added via os.add_dll_directory()
- Qwen3 thinking tags `<think>...</think>` stripped from output

**Done (Phase 2 — features):**
- **Meeting context**: textarea on upload page + supporting document attachments (DOCX/XLSX/PDF/TXT)
  - Text extracted from files via openpyxl/python-docx/PyPDF2 (CPU-only, no VRAM)
  - Context injected into summarization system prompt (~500 tokens max)
  - New field: `MeetingJob.context`, new Form params: `context`, `context_files`
- **Speaker renaming UI**: click speaker name in legend → inline rename → calls PUT /speakers/rename
  - Updates all segments globally in one API call
- **Re-summarize button**: saves edited transcript + triggers POST /resummarize
  - Loads LLM, re-runs summarization on edited transcript, regenerates DOCX/JSON
  - Navigates to processing page to show progress
- **Protocol page rewrite**: fetches actual MeetingProtocol JSON via GET /protocol/data
  - Shows: topic, summary paragraph, participants with speaking time % bars, transcript
  - No longer just shows raw segments
- **New API endpoints**:
  - `GET /api/meetings/{job_id}/protocol/data` — protocol JSON for frontend rendering
  - `POST /api/meetings/{job_id}/resummarize` — re-run summarization on edited transcript
- **Frontend polish**:
  - Elapsed time counter on ProcessingPage ("3:42 elapsed")
  - Logo clickable → navigates to /meetings
  - Dead App.css deleted
  - Ctrl+S keyboard shortcut to save transcript
  - Upload page messaging fixed
- **New dependencies**: openpyxl==3.1.5, PyPDF2==3.0.1

**Done (Phase 3 — bug fixes and polish):**
- Bug 43: Re-summarize crash — `detect_language()` passed string instead of segment list
- Bug 44: Transcript save wrote to `aligned_edited` but load read `aligned` — edits invisible
- Bug 45: DOCX not regenerated after speaker rename — now auto-regenerates
- Bug 46: `<think>` tags leaked into DOCX title — handles both closed and unclosed tags
- Bug 47: 56 micro-segments after postprocessor fix — restored speaker-turn merging (gap < 3s)
- Bug 48: Elapsed timer drifted — now uses Date.now() wall-clock reference
- Bug 49: Structured extraction echoed template placeholders — removed numbered examples from prompt
- Bug 50: Capitalization at segment joins — lowercase next segment start when no preceding punct
- `/no_think` directive added to all Qwen3 prompts to prevent thinking tokens consuming budget
- Anti-hallucination prompt strengthened for decisions/tasks extraction
- Claude API comparison button on protocol page (online only, requires ANTHROPIC_API_KEY)
- Debug logging for meeting context flow (upload → orchestrator → summarization)
- .gitattributes added for consistent LF line endings

**Offline test: PASSED (2026-04-08)**
- Full pipeline ran on ASUS ROG Zephyrus G14, RTX 4060 8GB, no network
- All models loaded from local cache (Whisper, SpeechBrain, Qwen3-8B GGUF)
- START_VERITAS_MVP.bat launches full stack with one click

**Known issues (carry forward):**
- MeetingsPage only shows in-memory jobs (not completed jobs from disk)
- Qwen3-8B occasionally halluccinates decisions (improved but not eliminated)
- GPU has headroom — consider context_size=8192 or Q5_K_M quant in future
- Russian-finetuned Whisper not yet tested vs vanilla
- No frontend tests

## Future Roadmap (Blocks 6–12)

### Block 6: Speaker Voice Enrollment
- Voice enrollment system for ~500 EPAM staff members
- Record 30-60 second voice sample, store encrypted voiceprint (embedding) on-premise
- Auto-identify speakers by name in meetings instead of "Speaker 1, Speaker 2"
- Quick enrollment for external participants (clients) before meetings
- "Unknown Speaker" handling with post-meeting enrollment option
- Voiceprint management UI: add, update, delete voice profiles
- All voiceprints encrypted at rest, never leave the machine

### Block 7: Live Transcription Mode
- Real-time transcription during meetings (live subtitles)
- WebSocket streaming from microphone to ASR engine
- Instant speaker identification using enrolled voiceprints (Block 6)
- Live dashboard for meeting chair: who is speaking, running transcript, timestamps
- Automatic transition to full pipeline after meeting ends (summarization, protocol)
- VRAM management: ASR model stays loaded during entire meeting session
- Fallback to post-meeting processing if live mode encounters issues

### Block 8: Legal Intelligence
- Action item extraction: automatically identify who promised what by when
- Decision tracking: tag and index every decision made in meetings
- Legal terminology enhancement: custom vocabulary layer for Russian legal terms
- Case linking: tag meetings to specific cases/matters
- Cross-meeting search: "when did we discuss X?" across all past transcripts
- Decision history: track how decisions evolved across multiple meetings on same topic

### Block 9: Smart Productivity
- Auto-generated meeting summary emails (protocol to participants)
- Full-text search across all meeting transcripts and protocols
- Meeting comparison: what changed between meetings on same topic
- Speaker statistics: talk time distribution, participation metrics
- Confidence highlighting: low-confidence words marked for manual review
- Meeting analytics dashboard: trends, frequency, duration over time

### Block 10: Security & Compliance Advanced
- Automated redaction for external sharing (client names, case numbers, financials)
- Role-based access control (RBAC): juniors see only their cases, partners see all
- Complete access audit trail: who viewed which transcript and when
- Configurable data retention policies with automated enforcement
- Export integration points for case management systems

### Block 11: HR Intelligence — Interview Analysis
- Record and analyze job interviews using the full VERITAS pipeline
- INTERNAL report (confidential, for interviewers):
  - Communication clarity and structure assessment
  - Technical depth indicators (surface buzzwords vs. genuine understanding)
  - Confidence and engagement patterns with transcript evidence
  - Structured scorecard with quotes as supporting evidence
  - Available within minutes of interview ending
- EXTERNAL report (for candidate, EPAM-branded PDF):
  - Communication profile and presentation style feedback
  - Speech pattern awareness (filler words, pacing, clarity)
  - Strongest moments highlighted (positive reinforcement)
  - Constructive development recommendations
  - NO scores, NO pass/fail, NO confidential assessments
  - Purely developmental — a gift to every candidate regardless of outcome
- Market differentiator: "EPAM gives back to every candidate"

### Block 12: VERITAS Care — Staff & Client Wellbeing
- Transparent, opt-in wellbeing intelligence system
- All participants informed that meeting analysis supports wellbeing
- STAFF CARE:
  - Meeting load tracking: flag burnout risk (too many hours in meetings)
  - Engagement trends: gradual drop in participation may signal burnout
  - Voice mood baseline: compare speaking patterns to personal norm over time
  - Workload balance visibility for team leads and HR
  - Gentle notifications to mentors/HR: "consider checking in with X"
- CLIENT CARE:
  - Relationship temperature: tone and engagement trends across meetings
  - Early warning for partners: "relationship with Client X may need attention"
  - Client satisfaction signals over time
- PRINCIPLES:
  - Fully transparent — staff and clients know the system exists
  - Opt-in where possible, clearly communicated where standard
  - Never used for punishment or secret surveillance
  - Trends and patterns only, never single-meeting judgments
  - Human decision-making always — AI provides signals, people act

## Integration: EPAM OS

VERITAS is developed as a standalone app but will integrate into EPAM OS (CRM, billing, client management) in the future. Architecture must stay modular and API-first. Auth system must be replaceable with SSO. Data IDs must be linkable to external CRM records. Do not build integration now, but do not make decisions that block it later.

## Bugs Fixed

### Block 2

1. **Orchestrator config mismatch**: NeMoASREngine and DiarizationEngine expected
   full AppConfig but received sub-configs (ASRConfig/DiarizationConfig). Fixed
   orchestrator to pass self._config instead of self._config.asr/diarization.
2. **DOCX table cell shading**: Used `get_or_add_pPr()` on CT_Tc element instead
   of `get_or_add_tcPr()`. This crashed when generating participant tables.
3. **Spectral clustering NaN**: Cosine similarity matrix had NaN values due to
   unnormalized embeddings. Fixed with L2 normalization + nan_to_num + clamping.
4. **Pydantic v2 deprecation**: `@validator` replaced with `@field_validator`,
   `Field(env=...)` replaced with `json_schema_extra`.
5. **Summarization import fragility**: `import torch` at top level without
   try/except (inconsistent with other engines). Added conditional import guards.
6. **CUDA guard in unload**: `torch.cuda.empty_cache()` called without
   `is_available()` check in summarization engine.

### First Real Run (post-Block 5)

7. **ffprobe `-print_json` unsupported on old Windows**: `audio.py` now tests ffprobe
   capabilities at startup and falls back to torchaudio for resampling/conversion.
8. **SpeechBrain symlink `[WinError 1314]`**: `diarization.py` monkey-patches
   `link_with_strategy` to copy files instead of creating symlinks on Windows.
9. **Progress callback arity mismatch**: Engines call `callback(float, str)` but
   orchestrator only accepted `(float)`. Added `message=""` default to all callbacks.
10. **TranscriptionSegment not subscriptable**: Diarization expected dicts but got
    Pydantic objects. Added `.dict()` conversion in orchestrator before diarization.
11. **Summarization decoded full prompt**: `_generate()` decoded entire sequence
    including input prompt. Fixed to decode only `outputs[0][input_length:]`.
12. **Progress bar math**: Stage multipliers were `* 0.25` instead of `* 25.0`.
    Fixed all three stage progress calculations in orchestrator.
13. **Endpoints fail after restart**: In-memory job state lost on restart. Transcript
    and protocol endpoints now check files on disk as fallback.
14. **Prompt artifacts in protocol tasks**: Summarization parser now filters known
    prompt header phrases from task extraction.

### Block 5c Fixes

15. **All ML engines tried to reach HuggingFace on every load**: Added offline-first
    loading to all 5 engines (SpeechBrain, Qwen, faster-whisper, NeMo, Silero VAD).
    Models load from local cache without any network access after initial download.
16. **SpeechBrain missing custom.py**: `from_hparams(source=local_dir)` requires
    `custom.py` even if empty. Created empty file + hyperparams.yaml path patching.
17. **HuggingFace transformers retry noise offline**: Even with `HF_HUB_OFFLINE=1`,
    `from_pretrained()` still tried HTTP requests. Fixed with `local_files_only=True`.
18. **Protocol download "Файл недоступен"**: Download used `<a href>` which sends no
    JWT token. Changed to fetch+blob download with Authorization header.
19. **Diarization overestimates speakers on short audio**: Eigenvalue gap heuristic
    unreliable with few embeddings. Replaced with silhouette score validation across
    candidate cluster counts.
20. **Transcript viewer "not found" after pipeline completes**: `aligned.json` was
    deleted by cleanup.py. Removed from cleanup list + added fallback to read
    transcript from protocol JSON when intermediate file missing.

### Block 5c Quality Improvements

21. **Filler words polluting transcript and protocol**: Added rule-based
    `postprocessor.py` that removes 25+ Russian and 14 English filler words,
    cleans repeated phrases (Whisper artifacts), and normalizes capitalization.
    Integrated between alignment and QA stages in orchestrator.
22. **Protocol in English for Russian meetings**: Rewrote all summarization
    prompts with Russian-language system/user messages. DOCX formatter now
    uses Russian section headers (УЧАСТНИКИ, РЕШЕНИЯ, etc.) and DD.MM.YYYY dates.
    Parser handles both RU and EN headers for robustness.
23. **Results buried in UUID directories**: Added archiving step that copies
    DOCX, JSON, and transcript to `data/archive/YYYY-MM-DD_filename/` for
    easy user access without knowing internal job IDs.
24. **Cleanup test assumed aligned.json deleted**: Test expected `aligned.json`
    to be removed by cleanup, but it is now intentionally preserved. Fixed
    assertion to expect the file to exist.

### Block 5d Fixes

25. **Summarization silently truncated 33% of input**: `max_length=8000` in
    tokenizer but `chunk_size=12000` in config. Each chunk lost ~4000 tokens.
    Fixed to use `config.summarization.chunk_size` as `max_length`.
26. **Silhouette validation used wrong clustering algorithm**: Speaker count
    estimation used AgglomerativeClustering but final clustering used
    SpectralClustering. Fixed to use same algorithm for both validation and
    final clustering. Also added preference for more speakers when scores close.
27. **Chunk overlap calculated by lines, not tokens**: `overlap_size` was a line
    count, but a single line could be hundreds of tokens. Fixed to calculate
    overlap in tokens, consistent with chunking logic.
28. **CPU compute type `"int8"` invalid for CTranslate2**: Only `float32`,
    `int8_float32`, `int8_float16` are valid standalone CPU types. Fixed to
    `"int8_float32"` for CPU. Also fixed: config `compute_type` was ignored
    (hardcoded `"float16"`), now uses config value for GPU.
29. **Aligner fell back to first speaker instead of closest**: When no time
    overlap found between transcription and diarization segment, fell back to
    `diarization[0]` (first speaker). Fixed to find nearest segment by time
    distance to the transcription segment midpoint.

### Runtime Bugs (On-Premise Testing, 2026-04-08)

30. **CTranslate2 CUDA deadlock on Whisper unload**: `self._model = None` triggers
    `__del__` synchronously, which deadlocks on CUDA synchronization in the
    asyncio thread. Segfaults if `unload_model()` called first (destructor tries
    to clean already-unloaded CUDA context). Fixed with parking pattern: call
    `unload_model()` to free GPU memory, then park reference in class-level
    `_parked_models` list so `__del__` never fires. (`whisper_asr.py`)
31. **VRAM leak — all engine unloads had gc/empty_cache in wrong order**: Every
    engine did `torch.cuda.empty_cache()` before `gc.collect()`. This means
    tensors freed by GC never had their CUDA cache cleared. Fixed all 4 engines
    (VAD, diarization, diarization embedder, summarization) to: `model.cpu()` →
    `del model` → `gc.collect()` → `torch.cuda.empty_cache()`.
32. **VRAMManager release_all() could deadlock**: Had `torch.cuda.synchronize()`
    call which can hang in same scenarios as CTranslate2. Removed — `empty_cache()`
    is sufficient.
33. **transformers 5.x incompatible with bitsandbytes 0.49.2**: `_is_hf_initialized`
    argument error. Fixed by pinning transformers==4.57.0 (supports Qwen3 >=4.51.0
    and is compatible with bitsandbytes 0.49.2).
34. **Qwen3-8B OOM during generate()**: 4-bit weights (5.96GB) + KV cache (~4.8GB)
    = 10.75GB, exceeds 8GB VRAM. Previous successful run used Qwen2.5-7B (4.5GB).
    Fixed by switching to Qwen3-4B (~2.5GB 4-bit), leaving ~5.5GB for inference.
    Config, summarization.py, and download script all updated.
35. **Summarization used device_map="auto" with CPU offload for Qwen3-4B**: Unnecessary
    since 4B model fits entirely on GPU. Switched to `device_map={"": 0}` which
    forces all layers onto cuda:0, avoiding CPU offload complexity.
36. **Config YAML never loaded**: `main.py` looked for `settings.yaml` in cwd, but
    file lives at `config/settings.yaml`. All config values used hardcoded defaults.
    Fixed to check `config/settings.yaml` first, then `settings.yaml` as fallback.
37. **settings.yaml initial_prompt was Latin transliteration**: Would replace correct
    Cyrillic default with garbled "Protocolul soveshchaniya". Removed from YAML so
    config.py Cyrillic default is used.
38. **Summarization OOM on generate()**: chunk_size=12000 + max_new_tokens=4096 = 16K
    total tokens needed ~4.7GB KV cache, exceeding 3.96GB free after model load.
    Fixed: chunk_size=3500, max_new_tokens=1500, total=5000 tokens (~1.7GB KV).
    Added gc.collect() + empty_cache() before every generate() call.
39. **Consolidation pass truncated to chunk_size**: When combining 3 chunk summaries
    (~4500 tokens), _generate() truncated input to chunk_size=3500, producing empty
    output. Fixed: _generate() now accepts max_input_tokens/max_new_tokens overrides.
    Consolidation uses input=4000, output=1000, total=5000 (same safe budget).
40. **Orchestrator retries useless after OOM**: No VRAM cleanup between retry attempts.
    Fragmented CUDA memory guaranteed cascade failure. Fixed: gc.collect() +
    empty_cache() between retry attempts.
41. **DOCX key topics duplicated**: Formatter rendered both title (truncated) and
    content (full text) for each TopicItem. Fixed to show content only as single bullet.

## Backup Strategy

- Folder backups: `VERITAS_backups/backup_blockN_YYYY-MM-DD/` before each block
- Git commits after each significant change within a block
- CLAUDE.md updated at end of each session

## Session Instructions for Claude

1. ALWAYS read this file first at the start of every session
2. Check git log for recent changes
3. Update "Current State" line above when completing work
4. **MANDATORY before session ends: update CLAUDE.md to reflect ALL changes made in this session.** This includes:
   - Architecture changes (new engines, new libraries, changed defaults)
   - New files added (update Backend/Frontend Structure sections)
   - New dependencies (update Dependencies section)
   - New or changed config options
   - Bug fixes (add to Bugs Fixed section)
   - Test count changes
   - Any changed behavior that a future session or strategy chat needs to know
   **This is critical because strategy sessions rely on CLAUDE.md for accurate project state. Stale docs cause wrong decisions.**
5. Commit to git and update this file before session ends
6. All code in English, no Cyrillic in source
7. Production-ready code only — no stubs, no placeholders
8. Security is non-negotiable — every feature must consider data privacy
9. Never trust CLAUDE.md blindly — if in doubt, verify against actual code with grep/read
