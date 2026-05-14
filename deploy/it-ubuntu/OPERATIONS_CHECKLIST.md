# VERITAS 1.0 IT Operations Checklist

## 1. One-time server preparation

- Ubuntu 22.04/24.04 LTS installed.
- NVIDIA driver installed.
- `nvidia-smi` shows the RTX GPU and VRAM.
- Docker Engine installed.
- NVIDIA Container Toolkit installed.
- Git installed.
- IT GitHub account has access to `https://github.com/vokgpt-cyber/Veritas`.
- Hugging Face read-token created.
- Access to `pyannote/speaker-diarization-community-1` accepted.

Useful checks:

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. Clone and first run

```bash
git clone --branch release/veritas-1.0-it https://github.com/vokgpt-cyber/Veritas.git
cd Veritas
chmod +x deploy/it-ubuntu/*.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

Edit the generated file:

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

## 3. Security gate before pilot users

- Read `deploy/it-ubuntu/SECURITY_REVIEW.md`.
- Keep `EPAM_FRONTEND_BIND=127.0.0.1` until reverse proxy/VPN/firewall rules are
  approved by IT.
- Do not publish backend `8765`, Ollama `11435`, or vLLM `8001` to the LAN.
- Change the bootstrap admin password before pilot use.
- Prefer `EPAM_AUTH_DEFAULT_PASSWORD_HASH` before non-local exposure.
- Keep `PYANNOTE_METRICS_ENABLED=0`.
- Do not commit `.env.server`, tokens, audio, transcripts, generated DOCX/JSON,
  audit logs, or model caches.
- Do not use images with `:latest`.
- Do not enable vLLM `--trust-remote-code` unless the exact model repository is
  reviewed and pinned.

Validate Compose and scan images:

```bash
docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml config --quiet

docker scout cves epam-veritas-backend:1.0
docker scout cves epam-veritas-frontend:1.0
docker scout cves ollama/ollama:0.13.4
docker scout cves vllm/vllm-openai:v0.18.2
```

Alternative scanner:

```bash
trivy image epam-veritas-backend:1.0
trivy image epam-veritas-frontend:1.0
trivy image ollama/ollama:0.13.4
trivy image vllm/vllm-openai:v0.18.2
```

Record any accepted vulnerabilities and compensating controls in the IT change
ticket.

## 4. VRAM allocation policy

Docker exposes GPU access. On GeForce RTX cards it does not reliably hard-slice
VRAM per container. Baseline Gemma 4 runs through Ollama. For future vLLM
experiments, the LLM VRAM budget is controlled through:

```bash
VERITAS_GPU_VRAM_BUDGET_GB=30
VLLM_GPU_MEMORY_UTILIZATION=0.62
VLLM_MAX_MODEL_LEN=65536
```

For RTX 4090 48 GB:

- 30 GB budget: `VLLM_GPU_MEMORY_UTILIZATION=0.62`
- 36 GB budget: `VLLM_GPU_MEMORY_UTILIZATION=0.75`
- 40 GB budget: `VLLM_GPU_MEMORY_UTILIZATION=0.83`

If vLLM fails with out-of-memory:

1. Confirm no other GPU-heavy process is running: `nvidia-smi`.
2. Lower `VLLM_MAX_MODEL_LEN`.
3. Lower model size.
4. Increase `VLLM_GPU_MEMORY_UTILIZATION` only if enough VRAM is available.

Do not switch VERITAS to a lower-quality fallback just to complete the run.

## 5. LLM selection

Baseline pilot model:

```bash
COMPOSE_PROFILES=
EPAM_SUMMARIZATION_PROVIDER=ollama
EPAM_SUMMARIZATION_OLLAMA_BASE_URL=http://ollama:11434
EPAM_SUMMARIZATION_OLLAMA_MODEL=gemma4:26b
OLLAMA_IMAGE=ollama/ollama:0.13.4
VERITAS_READ_ONLY_ROOTFS=false
```

Leave `VERITAS_READ_ONLY_ROOTFS=false` for the first deployment. If IT wants to
re-enable read-only root filesystems after smoke testing, verify torch/Triton
cache writes first.

First alternative candidate for later experiments only:

```bash
COMPOSE_PROFILES=vllm
EPAM_SUMMARIZATION_PROVIDER=openai_compatible
EPAM_SUMMARIZATION_OPENAI_BASE_URL=http://vllm:8000/v1
EPAM_SUMMARIZATION_OLLAMA_MODEL=Qwen/Qwen3.6-27B
VLLM_MODEL=Qwen/Qwen3.6-27B
VLLM_SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
```

When changing a vLLM model, update all three model-name fields together:

```bash
EPAM_SUMMARIZATION_OLLAMA_MODEL=<model_id>
VLLM_MODEL=<model_id>
VLLM_SERVED_MODEL_NAME=<model_id>
```

Then restart VERITAS.

Check Ollama:

```bash
docker logs -f veritas-ollama
docker exec veritas-ollama ollama list
```

Check vLLM experiments:

```bash
curl http://127.0.0.1:8001/v1/models
docker logs -f veritas-vllm
```

## 6. Updating VERITAS

```bash
cd Veritas
git pull --ff-only
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

The update rebuilds containers but keeps Docker volumes:

- meeting data volume;
- shared-data volume;
- Hugging Face cache;
- Torch cache;
- CTranslate2 cache;
- Ollama cache;
- vLLM cache.

## 7. Logs and diagnostics

All services:

```bash
docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml logs -f
```

Backend only:

```bash
docker logs -f veritas-backend
```

Ollama only:

```bash
docker logs -f veritas-ollama
```

vLLM only:

```bash
docker logs -f veritas-vllm
```

Container state:

```bash
docker ps
docker inspect -f '{{.State.Health.Status}}' veritas-backend
docker inspect -f '{{.State.Health.Status}}' veritas-vllm
```
