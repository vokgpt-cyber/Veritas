# VERITAS 1.0 Security Review for IT

This note is the security gate for the Ubuntu server package. VERITAS processes
law-firm recordings and documents, so the default stance is: local processing,
minimal network exposure, pinned runtime images, no silent cloud fallback, and no
arbitrary model code execution.

## What was hardened before handoff

- Frontend now runs as a production static build behind unprivileged Nginx, not
  the Vite development server.
- Frontend, backend, Ollama, and vLLM containers use `no-new-privileges`, drop Linux
  capabilities, and write model/runtime cache files only to explicit volumes or
  tmpfs mounts. `VERITAS_READ_ONLY_ROOTFS=false` is the first-deploy default
  because torch/vLLM/Triton need writable cache paths; IT may enable it after
  smoke testing.
- Server ports bind to `127.0.0.1` by default. Network access must go through an
  IT-approved reverse proxy, VPN, or firewall rule.
- Ollama and vLLM images are pinned to version tags instead of `latest`.
- `--trust-remote-code` is not enabled by default. Do not enable it unless the
  exact model repository has been reviewed as executable code.
- `ipc: host` was removed from the vLLM service.
- Backend upload handling now strips path components from uploaded file names,
  stores files under safe names, and enforces server-side upload size limits.
- Admin role checks no longer rely on the username `admin`; access requires the
  role stored in the user database.
- Python and frontend dependency audits were added to the release checklist.

## Required pre-production checks

Run these checks after building images and before exposing the service to pilot
users:

```bash
docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml config --quiet

docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml build

docker scout cves epam-veritas-backend:1.0
docker scout cves epam-veritas-frontend:1.0
docker scout cves ollama/ollama:0.13.4
docker scout cves vllm/vllm-openai:v0.18.2
```

If Docker Scout is not available, use Trivy or Grype:

```bash
trivy image epam-veritas-backend:1.0
trivy image epam-veritas-frontend:1.0
trivy image ollama/ollama:0.13.4
trivy image vllm/vllm-openai:v0.18.2
```

Do not approve production use while critical or high vulnerabilities remain in
internet-facing layers. For non-exposed ML runtime findings, IT should document
the accepted risk and compensating controls.

## Network policy

- Keep `EPAM_FRONTEND_BIND=127.0.0.1` until IT has placed VERITAS behind the
  approved access layer.
- Do not publish backend port `8765`, Ollama port `11435`, or vLLM port `8001`
  to the LAN.
- Allow outbound internet only during the model download window if possible.
  After model caches are populated, switch Hugging Face/Transformers offline
  flags to `1`.
- Keep `PYANNOTE_METRICS_ENABLED=0`.
- Do not configure cloud LLM endpoints for real law-firm data.

## Secrets policy

- `deploy/it-ubuntu/.env.server` must remain local to the server and must not be
  committed to Git.
- Use a read-only Hugging Face token with access only to required model
  repositories.
- Change the bootstrap admin password before any pilot user gets network access.
- Prefer `EPAM_AUTH_DEFAULT_PASSWORD_HASH` for server setup. If the launcher
  generates a temporary `EPAM_AUTH_DEFAULT_PASSWORD`, clear it after the admin
  password has been changed in the UI.
- Rotate `EPAM_AUTH_SECRET_KEY` and `EPAM_ENCRYPTION_SECRET_KEY` if `.env.server`
  is ever copied outside the server administration channel.

## Model supply-chain policy

Treat every model repository as a dependency. For the 1.0 pilot:

- Use Gemma 4 through Ollama as the pilot baseline.
- Use only the model IDs listed in `MODEL_GUIDE.md` unless IT approves an
  exception.
- Do not let pilot users add arbitrary Hugging Face model IDs.
- Do not enable `--trust-remote-code` for convenience. If a model needs it,
  review the repository code, pin the exact revision, document the decision, and
  run it in an isolated test environment first.
- Pin Docker images by tag at minimum. For stronger control, replace tags with
  IT-approved digests.

## Data and logs

- Docker volumes contain transcripts, protocols, voice samples, caches, and
  audit data. Back them up and protect them as confidential law-firm data.
- Developer log exports should contain technical diagnostics only, not full
  document contents.
- Generated DOCX, audio, transcripts, audit logs, model caches, `.env.server`,
  and tokens are excluded from the Git release package.

## Release verification performed by Codex

The release branch is checked for:

- no `.env.server`, `.env`, audio/video, DOCX, JSONL audit logs, model caches, or
  generated data directories;
- no Hugging Face/OpenAI/GitHub token patterns;
- frontend `npm audit` with zero reported vulnerabilities;
- Python `pip-audit` after dependency updates, excluding CUDA torch wheels that
  are not auditable through PyPI;
- Docker Compose syntax validation.

Final approval to connect VERITAS to the corporate network remains with EPAM IT.
