# VERITAS 1.0 Pilot Baseline

Release date: 2026-05-08

VERITAS 1.0 is the first pilot baseline for on-premise legal transcription,
court hearing stenograms, administrative meeting protocols, and action-list
drafting.

Security handoff update: the IT Ubuntu package now includes a production Nginx
frontend, local-only binds by default, Gemma 4 via Ollama as the pilot LLM
baseline, practical Docker write paths for torch/Triton caches, pinned
Ollama/vLLM image tags, no default
`--trust-remote-code`, safer upload handling, dependency CVE updates, and
`deploy/it-ubuntu/SECURITY_REVIEW.md`.

## Scope

- Windows workstation development remains in the project root.
- Local Docker startup remains `START_VERITAS_DOCKER.bat`.
- Legacy local MVP startup remains available for development diagnostics.
- IT server deployment files live in `deploy/it-ubuntu`.
- Application version exposed by `/api/health`: `1.0.0`.

## Quality Baseline

- ASR: GigaAM for Russian recordings, with no silent quality-reducing fallback.
- Diarization: pyannote `speaker-diarization-community-1`.
- Summarization: Gemma 4 26B via Ollama.
- Administrative protocols use the practical structure focused on participants,
  decisions, theses, and action items.
- Court hearings use exact speaker-count controls, case dictionary context, and
  speaker review workflow.

## Deployment Baseline

- Root Docker Compose is the Windows workstation compose.
- `deploy/it-ubuntu/docker-compose.server.yml` is the Ubuntu server compose.
- The Ubuntu server profile is prepared for RTX 4090 48 GB and optional vLLM.
- Model experiments are intentionally limited to the developer settings area.

## Important Policy

No low-quality fallback should be used just to complete a pipeline run. If the
configured quality path cannot run, VERITAS should fail clearly and let the
operator fix the environment, GPU memory, token, or selected model.
