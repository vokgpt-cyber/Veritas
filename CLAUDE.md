# EPAM VERITAS — Development State

**Verbal Intelligence, Transcription and Summarization**
On-premise meeting transcription and protocol generation for EPAM law firm.

## Current State: Un-engineering Sprint shipped 2026-05-04 (RTX 3090, GigaAM + Gemma 4 26B)

Production-readiness sprint phases A-E completed 2026-04-30. Then a SECOND sprint
(2026-05-04) rolled back the over-engineered admin protocol pipeline after user
feedback that map-reduce + heavy prompt + delete-based verification were producing
sparse protocols (2-3 decisions on hour-long meetings that had 15+). Industry research
(Otter, Fireflies, Notion, Teams, Anthropic Meeting Scribe) confirmed: every leading
meeting AI uses single-pass with direct prompts. We over-engineered defenses against
hallucination and ate our own content.

**What's live now (admin meeting pipeline):**
- Single-pass extraction with full transcript (map-reduce flag-gated, default OFF)
- Direct "extract structured protocol" prompt, no "you are a secretary" persona
- Implicit decisions count ("давайте перенесём" — это решение)
- LLM marks confidence high/medium/low per item in the FIRST call
- Verification pass DISABLED for admin (it was deleting legit content)
- Speaker resolution as separate post-pass: SPEAKER_X → attendee names via focused LLM call
- Low-confidence items shown with amber "[?]" tag in DOCX + UI; user curates

**What's pending validation:** A/B comparison harness exists at
`benchmarks/scripts/compare_pipelines.py` — runs OLD (map-reduce + heavy prompt +
verification-delete) vs NEW (single-pass + direct prompt + confidence-mark + speaker-
resolution) on the same archived transcript, generates side-by-side HTML. **Has NOT
been run yet on real archived data — that's the first verification step.**

## Architecture

- **Pipeline Order** (MVP): Preprocess -> ASR -> Diarization -> Align -> Post-process -> QA -> [Summarize] -> Format
  - This is the proven pre-5d order. Block 5d tried VAD pre-filtering and adaptive profiles but degraded quality — reverted.
  - Summarize stage is skippable via `summarization.enabled: false` (transcript-only mode)
  - Dead Block 5d code removed: VAD instantiation, `_select_profile()`, old structured protocol methods
- **ASR**: Five-engine, configurable:
  - **GigaAM v3_e2e_rnnt** (default, SOTA Russian): Sber's 240M-param Conformer, 700K hours Russian pretraining. WER 2.6-8.4% (vs Whisper 12-25%). ~3GB VRAM, ~250x realtime. Outputs punctuated, capitalized Russian directly. Uses Silero VAD for longform segmentation (pyannote VAD bypass due to torchcodec incompatibility). MIT license.
  - **HF Whisper** (Tier 2): `antony66/whisper-large-v3-russian` loaded via HuggingFace transformers pipeline in original format. ~3.5GB VRAM. Preserves fine-tuned model settings (avoids CTranslate2 conversion that breaks task config). Chunked + batched inference (batch_size=16, 30s chunks).
  - **faster-whisper** (Tier 3 fallback): `large-v3` or configurable HF repo, CTranslate2 with BatchedInferencePipeline for parallel GPU processing. Language pinned to `ru`.
  - **Qwen3-ASR** (parked, rejected): `Qwen/Qwen3-ASR-1.7B` — poor quality on Russian legal speech.
  - **NeMo** (Linux/Docker): NVIDIA NeMo Conformer (`stt_ru_conformer_transducer_large`)
  - Config: `engine: gigaam|hf-whisper|whisper|qwen|auto|nemo` in ASRConfig. Default: `gigaam`.
  - GigaAM installed from GitHub main branch (PyPI 0.1.0 only has v1/v2, GitHub has v3)
  - faster-whisper uses BatchedInferencePipeline (processes 16 chunks in parallel on GPU, ~30x speedup)
- **Audio Preprocessing**: Dual backend — ffmpeg (primary) with torchaudio fallback for old/missing ffmpeg
- **Diarization**: Dual-engine:
  - **pyannote-audio 4.0** (default): `pyannote/speaker-diarization-community-1`, VBx clustering (replaces agglomerative from 3.x), ~9.5GB VRAM, native min/max_speakers, state-of-the-art DER
  - **SpeechBrain** (fallback): ECAPA-TDNN + spectral/agglomerative clustering, silhouette scoring with same-algorithm validation, config-driven window/step
  - Config: `diarization.engine: pyannote|speechbrain`
  - pyannote 4.0 breaking changes: `use_auth_token` -> `token`, output is `DiarizeOutput` dataclass with `.speaker_diarization` attribute
- **Alignment**: Sum-aggregate overlap rule (winner = speaker with most summed diarization time inside the ASR window, not single-largest slice) + sentence-boundary pre-splitting at `.!?` using word-level timestamps. `AlignedSegment.attribution_confidence` = winner_time / total_diarized_time (surfaced in the side-by-side HTML as green/amber/red badge; <0.6 = contested).
- **Post-processing**: Punctuation restoration (deepmultilingualpunctuation, BERT-based, CPU, OFF by default — GigaAM native punct scored higher in the 2026-04-20 benchmark), filler removal (RU/EN), sentence cleanup, language detection. Same-speaker turn merge with gap-aware seam-polish: terminal punct → cap next; continuation punct → lowercase; no-punct + gap ≥ 0.6s → insert `.` + cap; no-punct + gap ≤ 0.15s → lowercase; ambiguous middle → leave alone. At every turn flush: leading em/en/hyphen + space dash marker stripped and first letter re-capitalized (regex requires whitespace after the dash, so internal dashes and compound words survive). `attribution_confidence` propagated as None-safe `min()` across merged children; ASR word confidence as duration-weighted avg.
  - Punctuation model: ~300MB, loads lazily on first use, no VRAM needed
  - Restores commas, periods, question marks that Whisper drops
- **Summarization**: Gemma 4 26B MoE (default, was T-Pro 2.0 until 2026-04-22 — see below) via Ollama HTTP API
  - Gemma 4 26B: ~16GB VRAM, 128K context, MoE architecture. Default summarization model
    since 2026-04-22 — disabled T-Pro 2.0 after observed hallucinations on admin meetings.
    Comments in `config/settings.yaml` and `backend/app/config.py:SummarizationConfig` keep
    T-Pro 2.0 referenced as a documented fallback (per user decision 2026-04-30) but the
    default value of `ollama_model` is `gemma4:26b`.
  - T-Pro 2.0 (parked, do NOT re-enable): `t-tech/T-pro-it-2.0:q4_K_M`, ~20GB VRAM, 32K
    context, #1 on Russian benchmarks (MERA 0.660, ruMMLU 0.790), Apache 2.0. Hallucinated
    on the admin meeting test corpus despite anti-hallucination prompt + verification pass —
    Gemma 4 has been more reliable on the same content.
  - Migrated from llama-cpp-python (Qwen3-8B GGUF) to Ollama HTTP client (httpx)
  - No more GGUF file management, CUDA DLL hacks, or manual VRAM management for LLM
  - Ollama manages its own GPU memory lifecycle — VRAMManager not involved
  - 128K context window: entire transcript in single pass (no chunking needed)
  - Full 7-section Russian legal protocol: Topic, Participants, Brief Summary, Key Discussion Points, Decisions, Action Items, Open Questions
  - Anti-hallucination: speaker citation requirements in prompt, verification pass (second LLM call fact-checks protocol against transcript), temperature 0.1, top_p 0.80
  - `summarization.enabled: false` skips LLM entirely for transcript-only DOCX output
  - Fallback models (kept documented but parked): `t-pro2:32b` / `t-tech/T-pro-it-2.0:q4_K_M` (Russian-optimized), `qwen3:32b` (multilingual). Switch via `ollama_model` config or `EPAM_SUMMARIZATION_OLLAMA_MODEL` env var.
- **Backend**: Python 3.11, FastAPI, WebSocket for progress + HTTP polling fallback
- **Auth**: JWT via python-jose + bcrypt password hashing
- **Frontend**: React 19 + Vite 8 + TailwindCSS 4 + lucide-react icons
- **Hardware**: RTX 3090 (24 GB VRAM), sequential model loading
- **Docker**: NVIDIA CUDA 12.1 base image, non-root user, hardened
- **Security**: AES-256-GCM encryption at rest, structured audit logging, auto cleanup

## Key Constraints

- **Privacy/Security**: CRITICAL. Law firm data. All processing on-premise, no data leaves the machine. No public code. No untrusted dependencies. Authentication required.
- **VRAM**: 24 GB (RTX 3090) — models load/unload sequentially via VRAMManager (except Ollama which manages its own VRAM)
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
    gigaam_asr.py    — GigaAM v3_e2e_rnnt ASR engine (default, Silero VAD + batched inference for longform)
    hf_whisper_asr.py — HuggingFace transformers Whisper engine (Tier 2, loads fine-tuned models in original format)
    qwen_asr.py      — Qwen3-ASR-1.7B ASR engine (alternative, with ForcedAligner timestamps)
    whisper_asr.py   — faster-whisper ASR engine (Tier 2 fallback), BatchedInferencePipeline for GPU parallelism
    diarization.py   — SpeechBrain ECAPA-TDNN diarization engine (fallback, config-driven)
    pyannote_diarization.py — pyannote-audio 4.0 diarization engine (default, VBx clustering)
    summarization.py — Gemma 4 26B summarization via Ollama HTTP API (7-section protocol)
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
  - `EPAM_ASR_ENGINE`: hf-whisper|whisper|qwen|auto|nemo (default: hf-whisper)
  - `EPAM_ASR_WHISPER_MODEL`: model name or HF repo path
  - `EPAM_ASR_CONDITION_ON_PREVIOUS_TEXT`: use previous chunk context (default: false)
  - `EPAM_ASR_NO_SPEECH_THRESHOLD`, `EPAM_ASR_COMPRESSION_RATIO_THRESHOLD`: Whisper tuning
  - `EPAM_DIARIZATION_ENGINE`: pyannote|speechbrain (default: pyannote)
  - `EPAM_DIARIZATION_WINDOW_DURATION`, `EPAM_DIARIZATION_STEP_DURATION`: sliding window params
  - `EPAM_SUMMARIZATION_ENABLED`: true/false — set false for transcript-only output (no LLM)
  - `EPAM_SUMMARIZATION_OLLAMA_BASE_URL`: Ollama API URL (default: http://localhost:11434)
  - `EPAM_SUMMARIZATION_OLLAMA_MODEL`: Ollama model tag (default: gemma4:26b)
  - `EPAM_SUMMARIZATION_OLLAMA_TIMEOUT`: request timeout in seconds (default: 300)

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
- qwen-asr — Qwen3-ASR-1.7B + ForcedAligner-0.6B (default ASR, Alibaba Qwen team)
- faster-whisper — CTranslate2 Whisper (Tier 2 ASR fallback, Guillaume Klein / SYSTRAN)
- nemo_toolkit — NVIDIA (ASR on Linux, optional on Windows)
- pyannote-audio — CNRS/IRIT (default diarization engine, pyannote 4.0 community-1)
- speechbrain — Mila / Cambridge (fallback diarization, SpeechBrain engine)
- gigaam — Sber GigaAM v3_e2e_rnnt ASR (default engine, MIT license, installed from GitHub main branch)
- Ollama — External LLM service for summarization (Gemma 4 26B default since 2026-04-22; T-Pro 2.0 parked but kept documented). No Python dependency — HTTP API via httpx.
- deepmultilingualpunctuation — CPU-based punctuation restoration for 47 languages incl. Russian (oliverguhr)
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

**Known remaining issues:**
- MeetingsPage only shows in-memory jobs (not completed jobs from disk)
- No frontend tests

## Block 6: SOTA Model Upgrade (RTX 3090) — IN PROGRESS

**Hardware migration**: RTX 4060 (8GB) → RTX 3090 (24GB). Codebase copied to new machine.

**Decision (2026-04-13)**: Replace all three ML engines with state-of-the-art models.

### Tiered Model Strategy

**Tier 1 (current default):**
- **ASR**: antony66/whisper-large-v3-russian via HFWhisperASREngine (~3.5GB VRAM) — Russian fine-tuned, WER 9.84%->6.39%. Loaded in original HF format (CTranslate2 conversion breaks it).
- **Diarization**: pyannote-audio 4.0 community-1 (~9.5GB VRAM) — best open-source speaker assignment
- **Summarization**: Gemma 4 26B MoE (A4B) via Ollama (~16GB VRAM Q4_K_M) — 128K context, MoE architecture, verification pass for anti-hallucination

**Tier 2 (fallback):**
- ASR: faster-whisper with Whisper large-v3 float16 + BatchedInferencePipeline
- Diarization: pyannote-audio 3.1
- Summarization: T-Pro 2.0 32B via Ollama (~20GB, Russian-optimized, custom Cyrillic tokenizer from T-Bank)

**Tier 3 (safe baseline):**
- ASR: faster-whisper Whisper large-v3 int8_float32 (pre-Block 6)
- Diarization: SpeechBrain (pre-Block 6 default)
- Summarization: Qwen3-32B via Ollama

**Parked (rejected by user):**
- Qwen3-ASR-1.7B — poor quality on Russian legal speech despite benchmark claims

### Implementation (2026-04-13)

**Step 1: Ollama + Gemma 4 Summarization — DONE**
- `backend/engine/summarization.py` completely rewritten: llama-cpp-python replaced with Ollama HTTP client (httpx)
- Removed: CUDA DLL hacks (`_setup_cuda_dll_paths`), GGUF file detection, torch imports, gc.collect/empty_cache, Llama class, tokenizer-based chunking
- `required_vram_gb = 0.0` (Ollama manages its own VRAM externally)
- `load()`: creates httpx.Client, verifies Ollama at base_url/api/tags, checks model availability
- `unload()`: sends `keep_alive: 0` to Ollama to release VRAM, closes httpx client
- `_generate()`: POST to /api/chat with `stream: false`, logs eval_count/prompt_count/total_duration
- `process()`: single-pass protocol generation (128K context, no chunking), then title generation, then section parsing
- `_build_protocol_prompt()`: full 7-section prompt in transliterated Russian with anti-hallucination instructions
- `_parse_protocol_sections()`: robust parser handling transliterated, Cyrillic Unicode, and English headers
- SummarizationConfig rewritten: removed `gguf_model_path`, `context_size`, `chunk_size`, `chunk_overlap`, `repetition_penalty`, `model`, `quantization`; added `ollama_base_url`, `ollama_model`, `ollama_timeout`
- Orchestrator: VRAM management removed around summarization stage (no register/unregister/release)
- `requirements.txt`: removed `llama-cpp-python`, added Ollama comment
- Formatter: all 7 protocol sections always rendered (not conditional)

**Step 2: pyannote-audio 4.0 Diarization — DONE**
- `backend/engine/pyannote_diarization.py` rewritten for pyannote 4.0 API
- `required_vram_gb` changed from 0.6 to 9.5
- `load()`: `use_auth_token` → `token` in from_pretrained kwargs
- `process()`: handles pyannote 4.0 DiarizeOutput — checks `hasattr(raw_output, "speaker_diarization")` with 3.x fallback
- DiarizationConfig defaults changed: `engine: "pyannote"`, `pyannote_model: "pyannote/speaker-diarization-community-1"`
- `requirements.txt`: `pyannote.audio==3.3.2` → `pyannote.audio>=4.0`
- `settings.yaml`: diarization section updated with new defaults

**Step 3: Qwen3-ASR-1.7B Transcription — DONE**
- New file: `backend/engine/qwen_asr.py` (~340 lines)
- Uses `qwen-asr` pip package: `Qwen3ASRModel.from_pretrained()` with ForcedAligner for word timestamps
- `required_vram_gb = 4.0` (1.7B ASR bf16 + 0.6B ForcedAligner bf16)
- `load()`: offline-first via HF_HUB_OFFLINE, loads both ASR and ForcedAligner on GPU
- `process()`: calls `.transcribe(audio, language, return_time_stamps=True)`, converts ForcedAligner timestamps to WordInfo
- `_group_words_into_segments()`: splits word-level timestamps into sentences by punctuation, pauses (>0.8s), or max duration (15s)
- `_get_language_name()`: maps ISO codes (ru, en) to full names (Russian, English) as required by Qwen3-ASR API
- ASRConfig: added `qwen_model`, `qwen_aligner_model` fields; default engine changed to `"qwen"`
- Orchestrator: added "qwen" engine option with fallback to faster-whisper if qwen-asr not installed
- `requirements.txt`: added `qwen-asr`
- `settings.yaml`: ASR section updated with qwen fields and engine default

### VRAM Budget (sequential loading on 24GB)
| Stage | Model | VRAM |
|-------|-------|------|
| ASR | antony66/whisper-large-v3-russian (HF pipeline) | ~3.5GB |
| Diarization | pyannote 4.0 community-1 | ~9.5GB |
| Summarization | Gemma 4 26B (Ollama) | ~16GB |

Peak VRAM at any stage: ~16GB. Comfortable margin on 24GB.

**Step 4: RTX 3090 Setup & Quality Fixes (2026-04-16) — IN PROGRESS**

Environment setup on new RTX 3090 machine:
- torch 2.8.0+cu126 installed (CUDA 13.0 driver, sm_86 architecture)
- torchcodec 0.7.0 + FFmpeg 7.1 shared libs + os.add_dll_directory() for pyannote 4.0
- pyannote-audio 4.0 community-1 model downloaded and verified working
- Whisper large-v3 cached and working
- antony66/whisper-large-v3-russian downloaded to `models/whisper-large-v3-russian/` (3.09GB, original HF format)
- Ollama + Gemma 4 26B running and verified

Bugs found and fixed on RTX 3090:
- Bug 51: faster-whisper VadOptions removed `threshold` param in 1.1+. Removed from whisper_asr.py.
- Bug 52: Ollama Gemma 4 thinking mode consumed entire token budget, returning empty content. Fixed: `"think": False` in request body + fallback to `thinking` field + `<think>` tag stripping.
- Bug 53: Summarization protocol prompt used transliterated Russian headers (KRATKOYE SODERZHANIYE). Gemma 4 couldn't parse them. Rewrote prompt to use English instructions + English section headers, content in Russian.
- Bug 54: Topic generation prompt also used transliterated Russian. Fixed to English instructions requesting Russian output.
- Bug 55: antony66/whisper-large-v3-russian model.bin couldn't load — HuggingFace symlinks broken on Windows despite Developer Mode. Reverted to standard large-v3 initially.
- Bug 56: CTranslate2 conversion broke antony66 model task configuration. Manual ct2 conversion lost the fine-tuned `task="transcribe"` setting, causing the model to translate Russian speech to English instead of transcribing. Even explicit `task="transcribe"` in faster-whisper didn't fix it. Solution: created new HFWhisperASREngine that loads model in original HuggingFace transformers format, bypassing CTranslate2 entirely.
- Bug 57: `'Segment' object has no attribute 'avg_log_prob'` — antony66 model segments lack this attribute. Fixed with `getattr(segment, "avg_log_prob", 0.0) or 0.0` in whisper_asr.py.

New ASR engine: HFWhisperASREngine (`backend/engine/hf_whisper_asr.py`):
- Loads Whisper models in original HuggingFace format via `WhisperForConditionalGeneration` + `WhisperProcessor`
- Preserves all fine-tuned model settings (task, language, etc.) — no CTranslate2 conversion needed
- **Rewritten for speed**: manual 30s chunking + batched `model.generate()` (replaces slow experimental `pipeline()`)
  - Previous approach: `pipeline("automatic-speech-recognition")` with `chunk_length_s=30` — 20+ min for 30-min audio, 23.7GB VRAM
  - New approach: split audio into 30s numpy chunks, batch through encoder+decoder via `model.generate(return_timestamps=True)`
  - Expected: ~1-3 min for 30-min audio on RTX 3090, ~7-8GB VRAM peak
  - `batch_size=8`: balances GPU parallelism vs VRAM (8 chunks × ~76MB working memory each)
- Timestamp parsing: `processor.tokenizer.decode(output_offsets=True)` with regex fallback for `<|0.00|>` tokens
- Loads from local `models/whisper-large-v3-russian/` directory, falls back to HuggingFace Hub
- Explicit `task="transcribe", language="ru"` in `model.generate()` to prevent translation
- `required_vram_gb = 3.5` (Whisper large-v3 in float16, `low_cpu_mem_usage=True`)

faster-whisper BatchedInferencePipeline (`backend/engine/whisper_asr.py`):
- Added `BatchedInferencePipeline` wrapper for parallel GPU chunk processing
- Processes 16 audio chunks simultaneously instead of sequentially (~30x speedup on RTX 3090)
- `batch_size=16` optimal for RTX 3090 with large-v3 float16 (~3GB model + ~8GB for chunks)
- Automatic fallback to sequential if BatchedInferencePipeline not available
- Added local directory resolution for model paths (checks project_root/model_size)

Quality improvements implemented:
- Punctuation restoration: `deepmultilingualpunctuation` (BERT-based, CPU, ~300MB) restores commas/periods/question marks that Whisper drops. Integrated into postprocessor before filler removal. Lazy-loaded on first use.
- Diarization temporal smoothing: `_smooth_diarization()` in orchestrator removes micro-segments (<150ms) and merges consecutive same-speaker segments with gaps <300ms. Reduces DER by 5-10%.
- Context injection verified: upload context + attached documents correctly reach LLM system prompt.

Anti-hallucination measures for Gemma 4 summarization (user reported ~40% hallucination):
- Temperature lowered from 0.3 to 0.2 (more deterministic/factual)
- Top-p lowered from 0.9 to 0.85 (narrower token distribution)
- Protocol prompt strengthened with speaker citation requirements: "When attributing a statement, you MUST name the speaker", "If you cannot find a speaker for a claim, DO NOT include it", "It is BETTER to write 'None' than to guess"
- Verification pass: new `_verify_protocol()` method sends a second LLM call that fact-checks every claim in the generated protocol against the original transcript, removing unsupported statements

Config changes:
- `asr.engine: "hf-whisper"` (HuggingFace transformers pipeline for antony66 model)
- `asr.whisper_model: "large-v3"` (hf-whisper engine loads from models/whisper-large-v3-russian/ directory directly)
- `asr.beam_size: 1` (greedy decoding, minimal quality loss, significant speed gain)
- `summarization.temperature: 0.2` (reduced from 0.3 for hallucination reduction)
- `summarization.top_p: 0.85` (reduced from 0.9 for hallucination reduction)
- `summarization.ollama_model: "gemma4:26b"`

New dependency: `deepmultilingualpunctuation==1.0.1`

**Step 5: GigaAM v3 + T-Pro 2.0 + torchcodec fixes (2026-04-17) — DONE**

Session replaced entire ASR stack and fixed critical Windows dependency issues:

Environment setup:
- GigaAM installed from GitHub main branch (PyPI 0.1.0 only has v1/v2 models, GitHub has v3 including v3_e2e_rnnt)
- Install command: `pip install --force-reinstall --no-deps https://github.com/salute-developers/GigaAM/archive/refs/heads/main.zip`
- gigaam pip package pins `torch<=2.5.1` but works fine with torch 2.8.0 (pip warns but doesn't block)
- T-Pro 2.0 Q4_K_M pulled via Ollama (`ollama pull t-tech/T-pro-it-2.0:q4_K_M`, 19GB)
- torchcodec fully uninstalled — incompatible with torch 2.8.0 on Windows, causes DLL load crashes
- HF_TOKEN added to .env for pyannote model downloads
- EPAM_DIARIZATION_ENGINE changed from speechbrain to pyannote in .env

New file: `backend/engine/gigaam_asr.py` (~350 lines):
- GigaAMASREngine extends BaseEngine with standard load/unload/process interface
- Short audio (<25s): uses GigaAM's base `transcribe()` method
- Long audio (>25s): custom Silero VAD + batched GigaAM inference path
  - Silero VAD segments audio into speech chunks (~1MB model, CPU, no HF token needed)
  - Chunks merged into 15-22s segments (30s hard max) matching GigaAM's input limits
  - Batched through GigaAM's encoder/decoder via model.forward() + model._decode()
  - This bypasses gigaam's built-in transcribe_longform() which depends on pyannote VAD → torchcodec (broken)
- Word-level timestamps enabled for both paths
- `required_vram_gb = 3.0`

Bug 59: torchcodec incompatible with torch 2.8.0 on Windows:
- torchcodec 0.7.0 AND 0.11.1 both crash with DLL errors on torch 2.8.0+cu126
- `aoti_torch_aten_narrow` not found in DLL, `_ARRAY_API` not found in onnxruntime
- pyannote.audio 4.0 imports AudioDecoder from torchcodec at module level; silently fails but crashes at runtime
- Fix: uninstalled torchcodec entirely, pre-load audio via torchaudio where needed
- GigaAM longform: replaced pyannote VAD with Silero VAD in engine
- Diarization: modified pyannote_diarization.py to pre-load audio as {"waveform": Tensor, "sample_rate": int} dict

Config changes:
- `asr.engine: "gigaam"` (was "hf-whisper")
- `summarization.ollama_model: "t-tech/T-pro-it-2.0:q4_K_M"` (was "gemma4:26b")
- ASRConfig default engine changed to "gigaam"

Orchestrator changes:
- Added "gigaam" engine option at top of selection chain
- Auto-detect now prefers GigaAM, falls back to whisper/nemo
- Import fallback: if gigaam not installed, falls back to faster-whisper

Test results (5-min Russian audio, `data/test/test_5min.wav`):
- GigaAM ASR: 18 segments in 32.4s, punctuated Russian output, clean load/unload
- pyannote diarization: 36 segments, 2 speakers in 4.4s, clean load/unload
- Both engines work independently; end-to-end pipeline test pending (needs real 30-60 min audio)

**Step 6: Post-processing A/B benchmarks (2026-04-20) — DONE**

Built A/B harnesses for every post-processing rule we were considering adding,
scored each against the `court_hearing_129` gold reference on matched word
pairs only (difflib alignment on normalised tokens; ASR word errors excluded
so we score the post-processing rule, not the ASR). Results are decisive.

Punctuation scoreboard (`benchmarks/reports/punct_scoreboard.json`):
- GigaAM native: micro F1 **0.732**, sentence-boundary F1 0.725
- deepmultilingualpunctuation: 0.537 / 0.575
- Silero ru_punct: 0.399 / 0.365

Capitalization scoreboard (`benchmarks/reports/cap_scoreboard.json`):
- GigaAM native (as_is): accuracy **94.6%**, cap-class F1 0.734, 296 mismatches
- sentence_caps_overlay (force cap after every .!?): 94.4% / 0.728 — slight regression
- oracle_proper_nouns (perfect NER from gold dict): 94.5% / 0.733 — also worse
- lower_then_sentence_caps (full lowercase + rule): 93.0% / 0.611

Top-15 cap offenders in both directions are common function words (и, у, мы, я,
то, это, по, в, а, ну, уважаемый, ответчик, том) — not proper nouns. Residual
cap errors are downstream of sentence-boundary disagreements, not missing NER.

Decisions locked in:
1. DMP and Silero ru_punct stay disabled on GigaAM/antony66 output (already
   enforced via `PostProcessingConfig.restore_punctuation=False`). The warning
   log explains the benchmark to future sessions.
2. No new capitalization normalizer, no Natasha NER pass. The existing
   per-segment `capitalize_sentences()` rule stays on — it's near-idempotent on
   GigaAM output and the flag is there to disable for no-caps engines.
3. The abbreviation map added in an earlier pass (`outputs/build_asr_diff.py`
   measurement units: млн/тыс/руб/кг/…) brought WER-hard 22.74% → 21.96% by
   moving 53 real matches out of the "substitution" bucket. Kept.

New scripts:
- `benchmarks/scripts/run_punctuation_variants.py` — generates DMP / Silero /
  gigaam_native text files (requires torch + models; runs on VERITAS machine).
- `benchmarks/scripts/score_punctuation.py` — pure Python scorer, per-class
  confusion + sentence-boundary F1.
- `benchmarks/scripts/build_capitalization_variants.py` — generates four
  capitalization variants (as-is, lower+rule, overlay, oracle).
- `benchmarks/scripts/score_capitalization.py` — pure Python scorer, binary
  capital-letter F1 + per-direction offender lists.

Postprocessor config change (`backend/app/config.py`):
- New `PostProcessingConfig` with env prefix `EPAM_POSTPROCESSING_`.
- Fields: `restore_punctuation=False`, `remove_fillers=True`,
  `remove_repetitions=True`, `capitalize_sentences=True`.
- `postprocessing` section added to `config/settings.yaml` and wired through
  `AppConfig.load_from_yaml`.
- Docstrings cite the 2026-04-20 benchmark numbers so future sessions don't
  re-enable DMP / add cap normalizers without reading the evidence.

Postprocessor runtime fix (`backend/core/postprocessor.py`):
- `postprocess_transcript()` previously ran DMP unconditionally on every
  segment. This was a live quality regression on GigaAM output (DMP F1 0.537
  overwriting GigaAM's 0.732). Signature now takes four independent flags
  (`restore_punctuation_enabled`, `remove_fillers`, `remove_repetitions`,
  `capitalize_sentences_enabled`), each gated at call site.
- Orchestrator reads `self._config.postprocessing` and passes the flags
  through. DMP is OFF by default; warning log explains why if re-enabled.

**Step 7: Aligner rewrite + postprocessor seam polish (2026-04-20 / 2026-04-21) — DONE**

Root cause found from the Админ 13-04 archive: the target interjection «Саш, не говори, пожалуйста, что это невозможно» (a SPEAKER_10 utterance at ~60.8s) was being attributed to SPEAKER_13 and buried inside a 20.7s mega-block. Two separate bugs combined to cause it.

Aligner rewrite (`backend/core/aligner.py`):
- Old rule: for each ASR segment, pick the diarization segment with the single largest time-overlap. On GigaAM's 15-22s longform chunks this picks whichever speaker happened to have the largest contiguous slice — even if a *different* speaker spoke more total words across the chunk.
- New rule: sum total overlap time per speaker across every diarization segment that intersects the ASR chunk. Winner = speaker with the most summed time. Matches perception ("who spoke the most inside this window").
- Sentence-boundary pre-splitting: before attribution, split each ASR segment at `.!?` using word-level timestamps, producing sub-segments with narrower acoustic windows. On the Админ file this took 194 raw segs → 849 fine segs. Short interjections that previously lived inside mega-blocks now get their own window.
- New field on `AlignedSegment`: `attribution_confidence: float | None` = winner_speaker_time / total_diarized_time inside the segment window. 0.94 = one dominant speaker, 0.5 = contested. Surfaced in the side-by-side HTML with green/amber/red badges; <0.6 flagged as contested.
- When no diarization overlaps the ASR window at all, fallback now picks the *temporally nearest* diarization segment by midpoint distance (was: `diarization[0]`, the first speaker). Bug 29 had been fixed for one path but this path slipped through.

Postprocessor seam-polish (`backend/core/postprocessor.py`):
- `merge_short_segments` now accepts `max_turn_gap_s: float = 3.0` and calls a new `_polish_seam(prev_text, next_text, gap)` helper at every merge seam. Four-branch rule:
  - Previous text ends in terminal punct (`.!?…`) → capitalize start of next.
  - Previous ends in continuation punct (`,;:—–-`) → lowercase start of next.
  - No punct + gap ≥ 0.6s (`GAP_SENTENCE_BREAK_S`, treated as audible full-stop) → insert `.` and capitalize next.
  - No punct + gap ≤ 0.15s (`GAP_CONTINUOUS_S`, mid-sentence) → lowercase next.
  - Ambiguous middle (0.15s < gap < 0.6s, no punct) → leave alone. ASR over-capitalizes segment starts; inserting artificial sentence breaks in this zone hurt more than it helped.
- `attribution_confidence` propagation: when merging N sub-segments into a turn, the merged segment's confidence is the None-safe minimum across children (so a single low-confidence slice marks the turn as contested). Previously it was dropped twice — once in `merge_short_segments` when reconstructing `AlignedSegment`, once in the per-segment text-clean loop in `postprocess_transcript`.
- ASR word-level confidence is now propagated as a duration-weighted average across merged children (was: unweighted or lost).

Verification on Админ 13-04 archive:
- Old aligner: 194 segments, avg 16.79s, longest 25.76s. Target phrase attributed to SPEAKER_13 inside a 20.7s block.
- New aligner (pre-polish): 849 segments, avg 3.50s, longest 16.99s. Target phrase isolated to [60.78–62.50] 1.72s SPEAKER_10 ac=0.94. SPEAKER_13's rebuttal at [62.66–68.63] 5.96s ac=0.91.
- New aligner + polished (reader-facing): 292 turns, avg 10.9s, longest ~186s. The long turn is a legitimate monologue at meeting wrap-up (same speaker, no other-speaker interleaving) — a readability concern (wall-of-text), not a correctness bug. Paragraph-break rule inside mega-monologues is parked for user decision.
- Attribution chain preserved through merges: the 1.72s isolation of the target phrase survives the same-speaker merge because it is bookended by SPEAKER_13 turns on both sides.
- 23 unit tests in `backend/tests/test_postprocessor.py` — `_polish_seam` (4 branches + empty inputs), `_strip_leading_speaker_dash` (em/en/hyphen, no-dash, dash-without-space, empty, re-cap, internal dash, single-strip), `merge_short_segments` integration (single turn strip, merged-same-speaker strip, speaker-change independent strips, min-attribution propagation, duration-weighted confidence), `_flush_turn` (no-op path, None buffer, short-turn drop). All 23 passing.

Leading speaker-dash strip (Phase A, 2026-04-21):
- `_LEADING_SPEAKER_DASH_RE = re.compile(r"^[\u2014\u2013\u002D]\s+")` in postprocessor.
- New `_strip_leading_speaker_dash(text)` strips em/en/hyphen + whitespace at the absolute start of a turn, then re-capitalizes. Internal dashes (direct-speech, compound words, ASR artifacts) are preserved because the regex requires whitespace after the dash.
- Applied via new `_flush_turn(merged, buffer)` helper at every turn boundary in `merge_short_segments` (mid-loop flush on speaker change AND tail flush on last buffer). Centralizes the flush logic so both paths get identical treatment.
- Drop-if-too-short check moved inside `_flush_turn` (previously only in per-segment loop).
- Verified on Админ 13-04 archive via new `benchmarks/scripts/repolish_alignment.py`: 849 aligned → 292 polished turns, 0 leading-dash turns remaining (previously 50+ visible in HTML).
- HTML report regenerated: 618KB, `<seg-text>` nodes starting with em-dash+space = 0 in polished column.

Phase A+ extensions (2026-04-21 second pass):
User review of the polished HTML surfaced five more turn-level pathologies. All resolved by widening the Phase A flush pass into a three-rule `_normalize_turn_text(text)` orchestrator:
1. **Leading-comma and dash+comma combinations** (e.g. ", в пятницу, наверное." or "—, там опять."). `_LEADING_SPEAKER_DASH_RE` widened to `r"^[\u2014\u2013\u002D,][\s,\u2014\u2013\u002D]+"` — the first char can now be a comma, and the trailing run accepts any mix of whitespace, commas, and dashes. Still requires at least one separator after the anchor char, so compound words ("IT-системы") and prefix dashes with no space ("—Да") survive.
2. **Internal direct-speech markers after sentence-end punct** (e.g. ". — Что было дальше" or "? —, Не слышу?"). When two same-speaker sub-segments merge, each sub-segment's own leading "— " ends up *inside* the turn right after the previous sub-segment's `.`/`?`/`!`/`…`. New `_INTERNAL_DASH_MARKER_RE = r"([.!?\u2026])\s+[\u2014\u2013\u002D][\s,\u2014\u2013\u002D]*(\S)"` and `_strip_internal_dash_markers()` strip those, re-capitalizing the first char of the new sentence. Legitimate grammar dashes (predicate-dash syntax: "дальше — неизвестно" where the dash follows a *word*, not punct) are preserved because the regex anchors on sentence-terminal punct.
3. **Repeated commas** (e.g. "А-а,, кое-как" left behind by filler removal + seam polish). New `_REPEATED_COMMA_RE = r",(?:\s*,)+"` and `_collapse_repeated_commas()` collapse ",," / ", ," / ",,," into a single ",".

Hesitation fillers added to `RU_FILLERS` (bare syllables that ASR transcribes with varying spellings between runs): а-а, аа, ааа, м-м, мм, ммм, э (bare), эм, э-э, ээ, эээ, ну-у. The `(?<!\w)X(?!\w)` boundary in `clean_filler_words` preserves substrings of legitimate words (коммерческий is not damaged by "мм").

`_flush_turn` now calls `_normalize_turn_text(text)` instead of `_strip_leading_speaker_dash(text)`. Rule order inside `_normalize_turn_text` is: internal-marker strip → comma collapse → leading strip+cap. Order matters because the leading-strip regex looks at the absolute turn start, which can change after the first two rules run.

Verification on Админ 13-04 archive (same `repolish_alignment.py`):
- 849 aligned → 291 polished turns (one turn shorter — a pure-comma turn was absorbed as too short after stripping).
- Rendered HTML polished-column audit (script inlined in session): 0 leading-dash turns, 0 leading-comma turns, 0 internal "`. — `" markers, 0 double-comma turns. Contrast: the NEW column (pre-polish) still has 74 leading-dash, 4 internal markers — so the three extensions actually do the work, the aligner alone is not enough.
- Five user-reported turn examples visually confirmed: "Не слышу? Не слышу? Алён? Понравился отчёт?" (clean), "В пятницу, наверное." (capital В, no leading comma), the 10:59-11:06 SPEAKER_05 turn now reads "Угу. Надо посмотреть, да. Да. А билто вы в итоге чё решили, как? Ничего. Там опять." (all dash markers gone).

Parked (still open): mid-sentence false capitalization of common nouns ("пройти Обследование") and spurious commas inside a phrase ("Я в пятницу, разговаривала") — both require LM/NER context to resolve safely, rules alone would do more harm than good.

Tests: `backend/tests/test_postprocessor.py` now 52 tests passing (was 23). New classes: `TestStripInternalDashMarkers` (8 tests covering period/question/dash-comma combo/grammar-dash preservation/ellipsis/empty/no-op/multi-marker chain), `TestCollapseRepeatedCommas` (7 tests), `TestNormalizeTurnText` (4 integration tests including the user's case #7 end-to-end), `TestHesitationFillers` (5 tests including the "коммерческий does not lose мм" guard). Extended `TestStripLeadingSpeakerDash` with leading-comma, dash-comma combo, double-comma prefix, and double-dash prefix cases. Updated one pre-existing test (double-dash prefix) because the new wider regex now intentionally strips both outer dashes rather than stopping at the first.

Side-by-side HTML report (`scripts/build_alignment_compare_html.py`):
- Three-column view: old aligner | new aligner | new + polished.
- Header stats: segment counts, avg/longest duration, attribution-confidence distribution.
- Speaker-share table with Δ (new − old) per speaker.
- Target-case callout with the phrase highlighted in red at top of page, ± context in each column.
- Color-coded confidence badges (green ≥0.8, amber ≥0.6, red <0.6), contested segments tinted in the body.
- Generated at `benchmarks/reports/alignment_compare.html` (~620KB, opens in browser).

Parked decisions (user to resolve):
- Wall-of-text paragraph-break rule for polished segments >~60s (e.g. insert soft paragraph break at sentence boundary after N seconds of same-speaker monologue).
- Confirm fix holds end-to-end in the live orchestrator on a fresh run of the Админ file (current verification is against the archived aligned_v2.json, not a fresh pipeline run).

### Remaining Block 6 Work
- **Phase 5: End-to-end test** — run full pipeline (ASR + diarization + alignment + summarization) on real 30-60 min Russian legal meeting. User will provide test file.
- **Phase 3: Subprocess isolation** (optional optimization) — run each VRAM phase in separate subprocess for guaranteed memory cleanup. Current VRAMManager sequential load/unload works but subprocess adds safety margin.
- **Phase 6: Summarization hardening** — verify T-Pro 2.0 anti-hallucination measures (temperature 0.2, speaker citations, _verify_protocol second-pass), implement GBNF grammar for structured JSON output
- Port anti-hallucination measures from Gemma 4 prompts to T-Pro 2.0 (should be LLM-agnostic)
- T-Pro 2.0 supports /think and /no_think directives — use /no_think in protocol generation to prevent thinking tokens consuming output budget
- Test engine fallback paths (gigaam->whisper, pyannote->speechbrain)
- Frontend: no changes needed (pipeline output format unchanged)

See `PROMPT_NEW_SESSION_RTX3090.md` for original Block 6 implementation instructions.

## Future Roadmap (Blocks 7–13)

### Block 7: Speaker Voice Enrollment
- Voice enrollment system for ~500 EPAM staff members
- Record 30-60 second voice sample, store encrypted voiceprint (embedding) on-premise
- Auto-identify speakers by name in meetings instead of "Speaker 1, Speaker 2"
- Quick enrollment for external participants (clients) before meetings
- "Unknown Speaker" handling with post-meeting enrollment option
- Voiceprint management UI: add, update, delete voice profiles
- All voiceprints encrypted at rest, never leave the machine

### Block 8: Live Transcription Mode
- Real-time transcription during meetings (live subtitles)
- WebSocket streaming from microphone to ASR engine
- Instant speaker identification using enrolled voiceprints (Block 6)
- Live dashboard for meeting chair: who is speaking, running transcript, timestamps
- Automatic transition to full pipeline after meeting ends (summarization, protocol)
- VRAM management: ASR model stays loaded during entire meeting session
- Fallback to post-meeting processing if live mode encounters issues

### Block 9: Legal Intelligence
- Action item extraction: automatically identify who promised what by when
- Decision tracking: tag and index every decision made in meetings
- Legal terminology enhancement: custom vocabulary layer for Russian legal terms
- Case linking: tag meetings to specific cases/matters
- Cross-meeting search: "when did we discuss X?" across all past transcripts
- Decision history: track how decisions evolved across multiple meetings on same topic

### Block 10: Smart Productivity
- Auto-generated meeting summary emails (protocol to participants)
- Full-text search across all meeting transcripts and protocols
- Meeting comparison: what changed between meetings on same topic
- Speaker statistics: talk time distribution, participation metrics
- Confidence highlighting: low-confidence words marked for manual review
- Meeting analytics dashboard: trends, frequency, duration over time

### Block 11: Security & Compliance Advanced
- Automated redaction for external sharing (client names, case numbers, financials)
- Role-based access control (RBAC): juniors see only their cases, partners see all
- Complete access audit trail: who viewed which transcript and when
- Configurable data retention policies with automated enforcement
- Export integration points for case management systems

### Block 12: HR Intelligence — Interview Analysis
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

### Block 13: VERITAS Care — Staff & Client Wellbeing
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

### Block 6 RTX 3090 Setup (2026-04-16)

51. **faster-whisper VadOptions threshold removed**: `threshold` parameter removed in
    faster-whisper 1.1+. Removed from vad_parameters in whisper_asr.py.
52. **Ollama Gemma 4 thinking mode**: Extended thinking consumed entire token budget,
    returning empty content. Fixed: `"think": False` in Ollama request body + fallback
    to read `thinking` field if `content` empty + regex `<think>` tag stripping.
53. **Summarization prompt transliterated Russian**: Gemma 4 couldn't parse transliterated
    Russian section headers (KRATKOYE SODERZHANIYE). Rewrote to English instructions with
    English section headers (SUMMARY, TOPICS, DECISIONS, TASKS, QUESTIONS), content in Russian.
54. **Topic generation prompt transliterated Russian**: Same issue as Bug 53.
    Rewrote to English instructions requesting Russian-language output.
55. **antony66/whisper-large-v3-russian symlink failure**: HuggingFace cache uses symlinks
    that break on Windows even with Developer Mode. model.bin unreadable. Fixed by
    downloading model to local `models/whisper-large-v3-russian/` directory via
    `snapshot_download()`, bypassing HF cache symlinks entirely.
56. **CTranslate2 conversion broke antony66 model task config**: Manual CTranslate2
    conversion of antony66/whisper-large-v3-russian lost the fine-tuned `task="transcribe"`
    setting. Model output English translations instead of Russian transcriptions. Even
    explicit `task="transcribe"` in faster-whisper didn't override it. Fixed by creating
    new HFWhisperASREngine (`hf_whisper_asr.py`) that loads model in original HuggingFace
    transformers format, preserving all model settings.
57. **'Segment' object has no attribute 'avg_log_prob'**: antony66 model's faster-whisper
    segments lack `avg_log_prob` attribute. Fixed with safe access:
    `getattr(segment, "avg_log_prob", 0.0) or 0.0` in whisper_asr.py.
58. **HFWhisperASREngine 40x too slow**: `pipeline("automatic-speech-recognition")` with
    `chunk_length_s=30` took 20+ minutes for 30-min audio and used 23.7/24GB VRAM on RTX 3090.
    The HF warning says chunking is "very experimental with seq2seq models." Rewrote engine
    to use manual 30s chunking + batched `model.generate(return_timestamps=True)` instead.
    Audio split into numpy chunks, batched through encoder+decoder (batch_size=8), timestamp
    tokens parsed from output. Expected: ~1-3 min, ~7-8GB VRAM. (`hf_whisper_asr.py`)

### Block 6 Step 5 — GigaAM + torchcodec (2026-04-17)

59. **torchcodec incompatible with torch 2.8.0 on Windows**: torchcodec 0.7.0 AND 0.11.1
    both crash with DLL errors (`aoti_torch_aten_narrow not found`, `_ARRAY_API not found`).
    pyannote.audio 4.0 imports `AudioDecoder` from torchcodec at module level — import
    silently fails but crashes at runtime when loading audio files. Affects both GigaAM
    longform (via pyannote VAD) and diarization. Fixed by: (a) uninstalling torchcodec
    entirely, (b) GigaAM engine uses Silero VAD instead of pyannote VAD for audio
    segmentation, (c) pyannote diarization engine pre-loads audio via torchaudio and
    passes `{"waveform": Tensor, "sample_rate": int}` dict instead of file path.

### Block 6 Step 7 — Aligner + postprocessor (2026-04-20 / 2026-04-21)

60. **Aligner max-single-overlap rule + no sentence splitting misattributed short
    interjections inside long ASR chunks**: Target phrase «Саш, не говори, пожалуйста,
    что это невозможно» (SPEAKER_10 at ~60.8s in the Админ 13-04 file) was buried
    inside a 20.7s block attributed to SPEAKER_13 — because SPEAKER_13 held the single
    largest contiguous diarization slice inside that GigaAM ASR chunk, even though
    SPEAKER_10 had more summed time across the chunk. Two-part fix in
    `backend/core/aligner.py`: (a) sum-aggregate overlap rule — winner = speaker with
    most summed time across all diarization segments intersecting the ASR window
    (was: single largest overlap); (b) sentence-boundary pre-splitting at `.!?` using
    word-level timestamps, producing narrower acoustic windows per attribution decision
    (194 → 849 sub-segments on Админ). New `attribution_confidence` field on
    `AlignedSegment` = winner_time / total_diarized_time, surfaced as green/amber/red
    badge in the side-by-side HTML. Also tightened: when no diarization overlaps at
    all, fallback now picks temporally nearest diarization segment by midpoint
    distance, not `diarization[0]` (Bug 29 had fixed one path; this path slipped).

61. **Postprocessor dropped attribution_confidence twice, and merges produced
    ugly seams**: `merge_short_segments` reconstructed `AlignedSegment` without
    passing through `attribution_confidence`, and the per-segment text-clean loop in
    `postprocess_transcript` did the same thing one call earlier. After the aligner
    started populating the field (Bug 60), it was wiped by the time the transcript
    reached the formatter. Second issue: when same-speaker sub-segments merged, the
    seam between them had inconsistent punctuation/capitalization (`"text one Text
    two"`). Fixed `backend/core/postprocessor.py`:
    (a) `merge_short_segments` now propagates `attribution_confidence` as None-safe
    `min()` across merged children (a single contested child marks the turn as
    contested), and propagates ASR word confidence as a duration-weighted average;
    (b) per-segment loop re-passes `attribution_confidence` when reconstructing
    segments; (c) new `_polish_seam(prev, next, gap)` helper applied at every merge
    seam — four-branch gap-aware rule: terminal punct → cap next; continuation punct
    → lowercase next; no-punct + gap ≥ 0.6s → insert `.` + cap; no-punct + gap ≤
    0.15s → lowercase next; ambiguous middle (0.15s < gap < 0.6s) → leave alone.
    Constants: `GAP_SENTENCE_BREAK_S=0.6`, `GAP_CONTINUOUS_S=0.15`. 11 unit tests
    covering all branches passing.

62. **Turns started with stray "— " direct-speech dash markers**: The Russian
    «тире» (em-dash + space) leaked into many polished turns in the side-by-side
    HTML — artifacts of speaker-change diarization boundaries and same-speaker
    merges colliding. User flagged this: all turns should begin directly with
    content, never with a dangling dash. Fixed `backend/core/postprocessor.py`:
    new `_LEADING_SPEAKER_DASH_RE = r"^[\u2014\u2013\u002D]\s+"` and
    `_strip_leading_speaker_dash()` helper strips the dash at turn-flush time
    and re-capitalizes the next character if it was lowercased (e.g. «— да.» →
    «Да.»). Regex requires whitespace *after* the dash, so internal dashes
    (direct-speech mid-turn, compound words like «IT-системы») are preserved.
    Strip is applied via new `_flush_turn()` helper at both flush points in
    `merge_short_segments` (speaker-change mid-loop and last-buffer tail). 12
    dedicated unit tests in `backend/tests/test_postprocessor.py` (class
    `TestStripLeadingSpeakerDash`, plus integration tests through
    `merge_short_segments`). Verified on Админ 13-04 archive: 849 aligned → 292
    polished turns, 0 leading-dash turns remaining in the regenerated HTML.
    Helper script: `benchmarks/scripts/repolish_alignment.py` regenerates
    aligned_v2_polished.json from aligned_v2.json without a full pipeline run.

63. **Polished turns still carried leading commas, internal "— " markers, and
    double commas after Bug 62**: User review of the Phase A HTML flagged five
    more turn-level pathologies that Bug 62's single-rule strip did not cover.
    (a) Turn starts with a stray leading comma ", в пятницу, наверное." (lower
    case, no capital), (b) turn starts with a dash+comma combo "—, там опять",
    (c) same-speaker merges carry each sub-segment's own leading "— " mid-turn
    right after the previous sub-segment's sentence-end punct (". — Что было
    дальше" / "? — Не слышу? — Алён?"), (d) double commas ",," left behind by
    filler removal ("А-а,, кое-как договорились"), (e) bare hesitation fillers
    (а-а, мм, э-э, ну-у) not in `RU_FILLERS`.
    Fixed in `backend/core/postprocessor.py` by widening Bug 62's single-rule
    pass into a three-rule `_normalize_turn_text()` orchestrator called from
    `_flush_turn`: (1) widen `_LEADING_SPEAKER_DASH_RE` anchor char class from
    `[\u2014\u2013\u002D]` to `[\u2014\u2013\u002D,]` and trailing run to
    `[\s,\u2014\u2013\u002D]+` so leading commas and dash+comma clusters are
    both caught; (2) new `_INTERNAL_DASH_MARKER_RE` and
    `_strip_internal_dash_markers()` — strip a dash+optional-commas+space
    sequence ONLY when it appears right after `.!?…` (never after a word, so
    predicate dashes like "дальше — неизвестно" survive), then re-capitalize
    the next char; (3) new `_REPEATED_COMMA_RE` and
    `_collapse_repeated_commas()` collapse ",," / ", ," into ",". Extended
    `RU_FILLERS` with hesitation variants (а-а, аа, ааа, а-а, м-м, мм, ммм, э,
    эм, э-э, ээ, эээ, ну-у). `(?<!\w)X(?!\w)` word-boundary in
    `clean_filler_words` preserves substrings of legitimate words (e.g.
    "коммерческий" is unchanged by the "мм" filler).
    Verified on Админ 13-04 archive via `benchmarks/scripts/repolish_alignment.py`:
    849 aligned → 291 polished turns. HTML polished-column audit: 0
    leading-dash, 0 leading-comma, 0 internal "— " markers, 0 double-comma
    turns. NEW column (pre-polish) still has 74 leading-dash and 4 internal
    markers — the three normalization rules actually do the work, aligner fix
    alone is not enough. Test count: 23 → 52 in `test_postprocessor.py` (new
    classes: `TestStripInternalDashMarkers`, `TestCollapseRepeatedCommas`,
    `TestNormalizeTurnText`, `TestHesitationFillers`; extended
    `TestStripLeadingSpeakerDash` with comma/combo/double cases). Two user
    pathologies remain parked: mid-sentence false capitalization of common
    nouns ("пройти Обследование") and spurious commas inside phrases ("Я в
    пятницу, разговаривала") — both require LM/NER context, not rule-safe.

### Production-Readiness Sprint Phase A (2026-04-30)

**Bugs 64–65 + new infrastructure for production deployment.**

64. **`backend/app/routes/meetings.py` syntax error after summarization
    rewrite**: Two issues compounded — (a) duplicate `@router.get`
    decorator + `async def get_transcript` block was pasted at lines
    585-594, breaking the docstring boundary because the second `def`
    landed *inside* the first `def`'s docstring; (b) 231 null bytes
    embedded around offset 0x0b420 of the file (file-write artifact
    from earlier session) silently broke Python's parser before it
    reached the user-visible code. Symptoms: `SyntaxError: invalid
    character '–'` at line 1097 (a misleading anchor — em-dashes inside
    docstrings ARE valid; the parser was misaligned by the duplicate
    block earlier in the file). Fixed by deleting the duplicate
    decorator+def block and stripping the null bytes. Verification:
    `python -c "import ast; ast.parse(open(...).read())"` returns OK,
    `grep -c "async def get_transcript"` returns 1.

65. **ProtocolPage shows 0/0/0:00 for admin meetings (regression from
    T3.3)**: Backend started returning admin protocols in the new
    wrapped shape `{meeting_type, payload}` (`backend/engine/protocols/
    schemas.py:ProtocolResult`) but the frontend still read
    `protocol.participants` and `protocol.transcript` at the top level
    (the legacy `MeetingProtocol` shape). Empty values rendered as
    zeros across the stats row and the body silently fell through with
    no error. Fixed in `frontend/src/types/api.ts` (added
    `WrappedProtocol` discriminated union and `isWrappedProtocol()`
    type guard mirroring backend dispatch) and `frontend/src/pages/
    ProtocolPage.tsx` (full rewrite: top-level `computeStats()` uses
    `payload.turns` for admin/court and `payload.epam_representatives
    + client_representatives` for client meetings; body dispatches to
    `AdminProtocolView` / `CourtProtocolView` / `ClientProtocolView` /
    `InterviewProtocolView` / `LegacyProtocolView` per `meeting_type`;
    each view typed against the corresponding payload interface). One
    file, ~1100 lines, no `any` casts. TypeScript `npx tsc --noEmit`
    clean.

**Pyannote court_hearing calibration (Task #31).** Court hearings
have multiple same-gender lawyers (judge + plaintiff's rep +
defendant's rep) whose voice embeddings VBx clustering tends to merge.
On `court_hearing_129` this produced a 59/33/8 turn-share split where
the two lawyers should have been roughly 35/35. Three knobs added to
`DiarizationConfig` (env prefix `EPAM_DIARIZATION_COURT_HEARING_*`),
applied ONLY when `meeting_type == "court_hearing"`:
- `court_hearing_min_segment_duration: 0.2` (down from global 0.5) —
  retains short interjections that signal speaker changes.
- `court_hearing_clustering_threshold: 0.55` (override pyannote
  community-1's pretrained ~0.7) — VBx splits clusters more
  aggressively, separating same-gender voices.
- `court_hearing_segmentation_min_duration_off: null` (default off,
  available for fast-paced cross-examination tuning).
Engine wires through `meeting_type` parameter on `process()` and
applies overrides via `pipeline.instantiate(...)` wrapped in
try/except (param-name mismatch logs warning and falls through to
defaults). Orchestrator passes `job.meeting_type.value`. SpeechBrain
fallback engine accepts the kwarg for API compatibility but ignores
it. Reset between meetings is automatic — pipeline reloads on each
job, so overrides don't leak.

**Backup of completed protocols (Task #32).** Replaced legacy
daily-grouped `data/archive/YYYY-MM-DD_filename/` with
month-grouped `data/backups/YYYY-MM/<filename>_<short-jobid>/`. New
`BackupConfig` at `backend/app/config.py` with env prefix
`EPAM_BACKUP_`: `enabled` (master toggle), `backup_dir` (root,
default `data/backups`), `retention_days` (None = keep forever,
positive int = prune older folders on each run). Job-id suffix
prevents same-filename collisions. Filename capped at 80 chars to
respect Windows MAX_PATH on deep mounts. Pruning is best-effort —
exceptions per folder logged at WARNING and don't stop the sweep,
empty month directories cleaned up after sweep. `_archive_results`
method retains its name for caller stability — internal layout
upgraded.

**CLAUDE.md sync (Task #33).** Top-of-file architecture line and
dependency description now correctly reflect Gemma 4 26B as the
default summarization model since 2026-04-22. T-Pro 2.0 references
preserved (per user decision 2026-04-30 — "Не убирай упоминания") as
documented fallback in YAML comments and code, but `ollama_model`
default is `gemma4:26b`.

### Un-engineering Sprint (2026-05-04)

**Trigger.** User feedback after running 3 admin protocols (#18 flat
9-section, #20 map-reduce topic-segmented): "слабовато по
административному совещанию". User compared with the EPAM IT team's
output using the SAME Gemma 4 26B model and a simpler pipeline —
their protocol was noticeably more useful. Investigation: I had
incorrectly assumed they used Gemini 2.5 Pro. They don't. They use
the same model with simpler architecture.

**Diagnosis.** Three layers of defense against hallucination
combined multiplicatively to suppress legitimate content:
1. Defensive prompt persona ("ты секретарь, не выдумывай, лучше
   вернуть меньше чем сомнительное")
2. Map-reduce that fragmented context per topic block (added 2026-04-30
   in task #35) — caused empty topic blocks when content was
   discussed across boundaries
3. Verification pass that DELETED items without explicit evidence

Independent industry research (Otter, Fireflies, Fathom, Granola,
Microsoft Teams Recap, Notion AI Meeting Notes, Google Meet "Take
Notes for Me", Anthropic Meeting Scribe) confirms 2025-2026 production
norm: single-pass with direct task prompt, light or no verification,
user curation post-hoc.

**What changed (tasks #43-47):**

* Task #43 — `summarization.use_topic_segmented_admin: false` is now
  the default. Map-reduce code path retained behind the flag for
  hypothetical 3+ hour meetings exceeding Gemma 4's effective attention.
* Task #44 — Admin prompt rewritten in
  `backend/engine/protocols/administrative.py:build_prompt`. Direct
  task instruction modeled on Anthropic Meeting Scribe + Notion AI.
  Implicit decisions ARE decisions: "давайте перенесём", "ну окей,
  делаем так" all count. owner / deadline are now permissive — empty
  allowed, user fills in post-hoc.
* Task #45 — `FlatProtocolItem` got a new `confidence: high|medium|low`
  field set by the LLM in the primary extraction call. Verification
  pass DISABLED for admin (`skip_verification = True` when
  `meeting_type == ADMINISTRATIVE` in `OllamaSummarizationEngine.process`).
  Other meeting types (court, client, interview) still run the
  verification pass — they have stricter accuracy requirements.
  DOCX renderer prefixes low-confidence items with "[?]" amber marker
  (`_render_flat_items_with_speaker`); UI shows amber-tinted card with
  "?" badge (`FlatItemList` in `ProtocolPage.tsx`).
* Task #46 — New module `backend/engine/protocols/speaker_resolution.py`.
  After protocol extraction, one focused Gemma call: "given attendees
  + sample turns per SPEAKER_X, return mapping". Apply across all
  speaker-bearing fields (decisions, tasks, open_questions, risks,
  topic_summaries inner fields, turns, participants). Unresolved
  SPEAKER_X get relabelled "Спикер #N" instead of staying as
  technical SPEAKER_06. Wired in `OllamaSummarizationEngine.process`
  via `_apply_speaker_resolution` immediately after `parse_response`.
* Task #47 — A/B harness `benchmarks/scripts/compare_pipelines.py`.
  Reads existing aligned.json + meeting context, runs OLD pipeline
  (map-reduce + heavy prompt + verification-delete) and NEW pipeline
  (single-pass + direct prompt + confidence-mark + speaker-resolution)
  on same input. Generates side-by-side HTML at
  `benchmarks/reports/compare_admin.html` plus raw JSON. Decision is
  data-driven, not gut-feel. **Not run on real data yet — that's
  the next verification step.**

**Court hearing table fill (2026-05-04).** User-requested cosmetic
fix at the same time: removed the "Light Grid" Word style from the
court_hearing stenogram table (caused subtle row banding) and replaced
with "Table Grid" (border-only). Added `_clear_cell_shading()` helper
that explicitly writes `<w:shd val="clear" fill="auto"/>` per cell to
override any conditional formatting from the style. Header cells keep
the EPAM-red shading (brand element).

**START_VERITAS_MVP.bat frontend port resilience.** User reported
that bat opens browser on `localhost:3000` while vite silently fell
back to `localhost:3001` because port 3000 was held by another
project. Fixed by: (a) pre-flight scan of ports 3000-3010 in the
launcher, (b) passing chosen port via `EPAM_FRONTEND_PORT` env var,
(c) `vite.config.ts` reads the env var with `strictPort: true` so
vite errors out instead of silently fallback. Browser-open uses the
resolved port.

**Files changed in un-engineering sprint:**
- `backend/app/config.py` — `use_topic_segmented_admin` default = false
- `config/settings.yaml` — same flag flipped
- `backend/engine/protocols/administrative.py` — `build_prompt` rewritten
- `backend/engine/protocols/schemas.py` — `FlatProtocolItem.confidence` added
- `backend/engine/protocols/speaker_resolution.py` — new module
- `backend/engine/summarization.py` — verification skip for admin,
  `_apply_speaker_resolution` post-pass added
- `backend/core/formatter.py` — `_clear_cell_shading`, court table
  uses Table Grid + per-cell clear, low-confidence "[?]" marker in
  `_render_flat_items_with_speaker`
- `frontend/src/types/api.ts` — `confidence` on `FlatProtocolItem`
- `frontend/src/pages/ProtocolPage.tsx` — `FlatItemList` shows amber
  badge + tinted background for low-confidence items
- `frontend/vite.config.ts` — `EPAM_FRONTEND_PORT` env support,
  `strictPort: true`
- `START_VERITAS_MVP.bat` — port 3000-3010 scan, frontend port passthrough
- `benchmarks/scripts/compare_pipelines.py` — new A/B harness

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
