# VERITAS 1.0 Ubuntu Server Deployment

This folder is the server handoff package for EPAM IT. Target environment:
Ubuntu 22.04/24.04 LTS, NVIDIA GPU, Docker, pyannote Community-1, GigaAM, and
vLLM/OpenAI-compatible summarization.

## Files

- `docker-compose.server.yml` - hardened server Compose file.
- `server.env.example` - example server environment.
- `START_VERITAS_UBUNTU.sh` - build/start launcher.
- `STOP_VERITAS_UBUNTU.sh` - stop launcher.
- `START_VERITAS_UBUNTU.desktop` - optional GUI launcher.
- `MODEL_GUIDE.md` - model and VRAM guide.
- `OPERATIONS_CHECKLIST.md` - operating checklist.
- `SECURITY_REVIEW.md` - required security checklist before pilot exposure.

## Prerequisites

Install on the server:

1. Ubuntu 22.04/24.04 LTS.
2. NVIDIA driver, verified by `nvidia-smi`.
3. Docker Engine.
4. NVIDIA Container Toolkit.
5. Git.
6. Hugging Face read-token with accepted access to
   `pyannote/speaker-diarization-community-1`.

GPU check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Clone

```bash
git clone --branch release/veritas-1.0-it https://github.com/vokgpt-cyber/Veritas.git
cd Veritas
```

For updates:

```bash
git pull --ff-only
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

## First run

```bash
chmod +x deploy/it-ubuntu/*.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

The launcher creates `deploy/it-ubuntu/.env.server` on first run, generates
random auth/encryption secrets, and generates a one-time bootstrap admin
password. It prints that password in the terminal.

Before the first real run, edit:

```bash
nano deploy/it-ubuntu/.env.server
```

Set:

```bash
HF_TOKEN=hf_...
```

Then restart:

```bash
./deploy/it-ubuntu/STOP_VERITAS_UBUNTU.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

## Network exposure

Default server package binds the web UI to localhost only:

```bash
EPAM_FRONTEND_BIND=127.0.0.1
EPAM_FRONTEND_PORT=5173
```

Do not publish backend `8765` or vLLM `8001` to the LAN. To give users access,
put the frontend behind an IT-approved reverse proxy, VPN, or firewall rule.

For non-local exposure, set a real admin password hash first:

```bash
EPAM_AUTH_DEFAULT_PASSWORD_HASH=<bcrypt hash>
EPAM_AUTH_DEFAULT_PASSWORD=
```

If the launcher-generated temporary password is used for bootstrap, change it in
the UI before pilot users get access.

## Security gate

Before pilot users:

```bash
docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml config --quiet

docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml build

docker scout cves epam-veritas-backend:1.0
docker scout cves epam-veritas-frontend:1.0
docker scout cves vllm/vllm-openai:v0.18.2
```

If Docker Scout is unavailable, use Trivy or Grype:

```bash
trivy image epam-veritas-backend:1.0
trivy image epam-veritas-frontend:1.0
trivy image vllm/vllm-openai:v0.18.2
```

Read `SECURITY_REVIEW.md` and record any accepted risks in the IT change ticket.

## GPU and VRAM

Docker exposes GPU access. On GeForce/RTX cards it usually does not hard-slice a
fixed VRAM amount per container. For LLM, VERITAS controls the budget through
vLLM:

```bash
VERITAS_GPU_VRAM_BUDGET_GB=30
VLLM_GPU_MEMORY_UTILIZATION=0.62
VLLM_MAX_MODEL_LEN=65536
```

For RTX 4090 48 GB:

- 30 GB budget: `0.62`
- 36 GB budget: `0.75`
- 40 GB budget: `0.83`

If the chosen model needs more memory than allocated, it should fail clearly.
Do not enable lower-quality fallback models just to complete the pipeline.

## LLM selection

Default server mode uses vLLM:

```bash
COMPOSE_PROFILES=vllm
EPAM_SUMMARIZATION_PROVIDER=openai_compatible
EPAM_SUMMARIZATION_OPENAI_BASE_URL=http://vllm:8000/v1
EPAM_SUMMARIZATION_OLLAMA_MODEL=Qwen/Qwen3.6-27B
VLLM_MODEL=Qwen/Qwen3.6-27B
VLLM_SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
```

When changing the model, update all three model-name fields together. Use only
approved model IDs from `MODEL_GUIDE.md` unless IT reviews an exception.

Do not enable `--trust-remote-code` unless the exact model repository has been
reviewed and pinned.

## Logs

All services:

```bash
docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml logs -f
```

Backend:

```bash
docker logs -f veritas-backend
```

vLLM:

```bash
docker logs -f veritas-vllm
```

## Stop

```bash
./deploy/it-ubuntu/STOP_VERITAS_UBUNTU.sh
```

This stops containers but keeps data, model caches, settings, and Docker
volumes.

## Official references

- Docker resource constraints: `https://docs.docker.com/engine/containers/resource_constraints/`
- Docker Compose services: `https://docs.docker.com/reference/compose-file/services/`
- NVIDIA Container Toolkit: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/`
- vLLM OpenAI-compatible server: `https://docs.vllm.ai/en/latest/serving/openai_compatible_server/`
- OWASP Docker Security Cheat Sheet: `https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html`
