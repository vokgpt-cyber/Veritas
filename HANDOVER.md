# VERITAS — Handoff to a new chat / new computer

**Project:** EPAM VERITAS — on-premise meeting transcription + protocol
generation for an EPAM law firm. Stack: GigaAM (ASR) → pyannote 4.0
(diarization) → Gemma 4 26B via Ollama (summarization). RTX 3090 (24GB).
Russian-language. JWT auth + AES-256-GCM at rest. FastAPI backend +
React frontend, launched via `START_VERITAS_MVP.bat`.

## START HERE — read this first

1. Read `CLAUDE.md` (bottom-of-repo, 1300+ lines) for the full project
   state, architecture, bugs fixed list, and roadmap. **It is
   authoritative.** This handoff doc is a TL;DR pointer, not a
   replacement.
2. Read recent sections in this order if you only have time for one:
   `## Current State`, `### Un-engineering Sprint (2026-05-04)`,
   `## Future Roadmap (Blocks 7–13)`.
3. Verify project state matches what `CLAUDE.md` claims — see the
   verification checklist below before assuming anything.
4. The user is non-technical (law firm partner). Don't dump shell
   commands without explanation. Default to plain prose with light
   structure.

## Where things live

| Path | What |
|---|---|
| `backend/app/main.py` | FastAPI lifespan, CORS, config load, user data root |
| `backend/app/config.py` | All Pydantic config classes |
| `backend/core/orchestrator.py` | Pipeline coordinator (8 stages) |
| `backend/engine/gigaam_asr.py` | Default ASR (Sber GigaAM v3) |
| `backend/engine/pyannote_diarization.py` | Default diarization |
| `backend/engine/summarization.py` | Ollama HTTP client + admin orchestration |
| `backend/engine/protocols/administrative.py` | Admin prompt + parser |
| `backend/engine/protocols/speaker_resolution.py` | SPEAKER_X → name post-pass |
| `backend/engine/voice_enrollment.py` | Block 7 — enrolled voice DB |
| `frontend/src/pages/ProtocolPage.tsx` | Renders all 5 meeting types |
| `frontend/src/pages/SpeakersPage.tsx` | Voice enrollment UI |
| `frontend/src/pages/AuditPage.tsx` | Admin-only audit log viewer |
| `config/settings.yaml` | All operator-tunable knobs |
| `config/hot_words.yaml` | ASR vocabulary biasing |
| `config/address_corrections.yaml` | Postprocessor regex dictionary |
| `START_VERITAS_MVP.bat` | One-click launcher |
| `benchmarks/scripts/compare_pipelines.py` | A/B harness — un-engineering sprint validator |

## Current state at handoff (2026-05-04 end of session)

**Most recent work — un-engineering sprint, tasks #43-47.** User
flagged that admin protocols felt "weak" compared to the EPAM IT team's
output (same Gemma 4 26B, simpler pipeline). Investigation showed we
had stacked three layers of defense against hallucination — defensive
prompt + map-reduce fragmentation + delete-based verification — that
combined multiplicatively to suppress legitimate content. Industry
research (Otter, Fireflies, Notion, Anthropic Meeting Scribe) all use
single-pass with direct prompt. We rolled back the defenses:

- Map-reduce default OFF (flag-gated for opt-in)
- Direct prompt without "secretary" persona
- Implicit decisions count ("давайте перенесём" = решение)
- Verification pass disabled for admin meetings; LLM marks confidence
  per item in the primary call instead
- Speaker resolution as separate post-pass (focused single-task LLM
  call mapping SPEAKER_X to attendee names)
- A/B harness built but **NOT YET RUN** on real archived data — that
  is the first thing to verify against the user's real meetings

**Validation pending.** The new pipeline has not been benchmarked
against the old. Run:
```
python benchmarks/scripts/compare_pipelines.py \
    --input "<path-to>/data/backups/2026-04/Админ_*/aligned.json" \
    --output benchmarks/reports/compare.html
```
If the new pipeline doesn't visibly win → investigate prompt /
consider model swap (Qwen 3 32B was researched as a candidate but not
chosen — Gemma 4 26B with simple pipeline expected to be enough).

## Critical decisions locked in (don't rediscuss)

- **Default summarization model: Gemma 4 26B via Ollama.** Do NOT
  re-enable T-Pro 2.0 (parked 2026-04-22 due to hallucinations on
  admin meetings). T-Pro stays referenced as documented fallback in
  YAML comments per user explicit decision: "Не убирай упоминания".
- **Default ASR: GigaAM v3_e2e_rnnt** (Sber, MIT license). Beats
  Whisper on Russian by 5-10pp WER on legal speech. Use HF Whisper
  (antony66/whisper-large-v3-russian) only when meeting has ≥50%
  English content.
- **Default diarization: pyannote-audio 4.0 community-1** with VBx
  clustering. Court hearings get `clustering_threshold: 0.55` override
  via `EPAM_DIARIZATION_COURT_HEARING_*` env vars to separate
  same-gender lawyer voices.
- **Per-user data isolation:** all user data lives at
  `%APPDATA%\VERITAS\<windows-username>\data\`. Voice DB (shared
  across users) at `%PROGRAMDATA%\VERITAS\voices.db`. Override via
  `EPAM_DATA_ROOT=data` to revert to legacy project-relative paths.
- **No cloud LLMs in production.** Cloud (Anthropic API via
  `summarize-cloud` endpoint) is allowed only as a comparison /
  testing button, not as a default summarization path.
- **Map-reduce protocol approach: rejected as default 2026-05-04.**
  Single-pass wins for meetings <2 hours that fit in 128K context.
  Code remains for hypothetical longer meetings.

## Verification checklist for the new chat

Before assuming anything in `CLAUDE.md` is current, verify:

```bash
# 1. Config has the expected defaults
grep "use_topic_segmented_admin" config/settings.yaml backend/app/config.py
grep "ollama_model" config/settings.yaml
grep "engine:" config/settings.yaml | head -3

# 2. New module exists and is sound
python -c "import ast; ast.parse(open('backend/engine/protocols/speaker_resolution.py').read())"

# 3. Verification pass actually skipped for admin
grep -n "skip_verification" backend/engine/summarization.py

# 4. Frontend port resilience in launcher
grep -n "EPAM_FRONTEND_PORT" START_VERITAS_MVP.bat frontend/vite.config.ts

# 5. Git status to see uncommitted work
git status
git log -10 --oneline
```

## Outstanding items (next session can pick up)

In rough priority order:

1. **Run A/B harness on real archived admin meeting** to validate the
   un-engineering sprint actually improved quality. If it didn't, dig
   into the prompt or consider Qwen 3 32B swap.
2. **Voice enrollment first-time setup.** User has the UI but hasn't
   enrolled employees yet. First enrollment requires `HF_TOKEN` in
   `.env` to download `pyannote/embedding` (~100MB, cached after).
3. **Hot-words dictionary refinement** — `config/hot_words.yaml`
   ships with the EPAM employees we know about; add new names as
   they appear in mis-transcriptions.
4. **Address normalization dictionary** — same pattern at
   `config/address_corrections.yaml`. User-editable YAML.
5. **Migration of existing data.** When the user upgrades from a
   project-relative `data/` install to the new per-user
   `%APPDATA%\VERITAS\<user>\` layout, old protocols don't auto-move.
   No migration script yet — they either set `EPAM_DATA_ROOT=data`
   to keep using the old folder or copy manually.

## Roadmap (Blocks 7–13, from CLAUDE.md)

- **Block 7 — Voice enrollment.** Backend + UI shipped 2026-04-30
  (tasks #36-38). Awaiting first real-world enrollment.
- **Block 8 — Live transcription.** Real-time transcription during
  meetings. Not started.
- **Block 9 — Legal Intelligence.** Action item extraction + decision
  tracking + cross-meeting search. Not started.
- **Block 10 — Smart Productivity.** Auto-summary emails, full-text
  search, meeting comparison. Not started.
- **Block 11 — Security & Compliance Advanced.** Redaction, RBAC,
  retention policies. Audit log viewer (Block 11 sneak preview)
  shipped 2026-04-30 as task #42.
- **Block 12 — HR Intelligence.** Interview analysis with internal
  scorecard + external developmental report. Not started.
- **Block 13 — VERITAS Care.** Wellbeing intelligence with explicit
  opt-in. Not started.

## Things the user is sensitive about

- **Style preferences.** User asks for plain prose, not bullet lists,
  unless asked. Russian for user-facing output. Don't dump shell
  commands without context.
- **"Не убирай упоминания T-Pro"** — keep T-Pro 2.0 referenced as
  documented fallback even though it's not used.
- **"Не нужно хардкодить"** — when proposing fixes, prefer
  data-driven / configurable solutions over hardcoded constants.
- **"Не гонись за бенчмарками"** — research what production systems
  actually do, not what scores best on MMLU.
- **"Я non-IT-guy, нужна кнопка"** — every operational feature should
  have a UI toggle / batch script wrapper. No "set this env var"
  workflows.
- **A/B test before committing to architectural changes.** Don't
  rebuild things hoping they'll be better — generate evidence.

## Session restart instructions

When the user starts a new chat in another computer or with a fresh
context:

1. They paste this `HANDOVER.md` content into the first message.
2. They ask the agent to read `CLAUDE.md` for full state.
3. They run the verification checklist above to confirm code matches
   docs.
4. They mention which area they want to work on (validation of un-
   engineering, voice enrollment first-run, hot-words refinement,
   etc.).

## File layout for transfer

If transferring to another machine, the minimum needed:

```
VERITAS 2/
├── backend/                 # Python source
├── frontend/                # React source (run `npm install` after copy)
├── config/                  # YAML configs (settings, hot_words, address_corrections)
├── benchmarks/              # Eval scripts including compare_pipelines.py
├── scripts/                 # Various .bat helpers
├── data/                    # User data (skip if going to use %APPDATA% per-user)
├── models/                  # Optional: pre-downloaded ML models
├── requirements.txt
├── CLAUDE.md                # MAIN reference — read this first
├── HANDOVER.md              # This file
├── COWORK.md                # User session notes (if present)
├── START_VERITAS_MVP.bat
├── REBUILD_VENV.bat         # Run on first install (~20 min)
└── .env                     # Secrets (HF_TOKEN, EPAM_AUTH_SECRET_KEY etc.) — EXCLUDE if checking into version control
```

**Don't transfer:**
- `venv/` — rebuild via `REBUILD_VENV.bat`
- `node_modules/` — `npm install` recreates
- `.git/` — only if continuing the same git history
- `data/meetings/<job_id>/` working dirs — these are temporary
- `models/` cache — re-downloads on first run if `HF_TOKEN` in `.env`

**Required external services on the target machine:**
- Python 3.11+
- Node.js 20+
- Ollama installed and running (`ollama serve`)
- Ollama model pulled: `ollama pull gemma4:26b` (~16GB, one-time)
- ffmpeg in PATH (or rely on the torchaudio fallback we built)
- NVIDIA driver supporting CUDA 12.1+ if using GPU
- HuggingFace token in `.env` (`HF_TOKEN=...`) for first download of
  pyannote and Whisper models
