#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.server"
ENV_EXAMPLE="${SCRIPT_DIR}/server.env.example"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.server.yml"

make_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

make_password() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 24 | tr -d '=+/' | cut -c1-24
  else
    head -c 24 /dev/urandom | base64 | tr -d '=+/' | cut -c1-24
  fi
}

BOOTSTRAP_PASSWORD=""

echo
echo "======================================================"
echo " VERITAS 1.0 Ubuntu server launcher"
echo "======================================================"
echo

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] Docker is not installed or not on PATH."
  echo "        Install Docker Engine and NVIDIA Container Toolkit first."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker daemon is not running or current user has no access."
  echo "        Start Docker or add the user to the docker group."
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi was not found."
  echo "        Install NVIDIA driver before running VERITAS."
  exit 3
fi

if [ ! -f "${ENV_FILE}" ]; then
  echo "[SETUP] Creating ${ENV_FILE}"
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  AUTH_SECRET="$(make_secret)"
  ENC_SECRET="$(make_secret)"
  BOOTSTRAP_PASSWORD="$(make_password)"
  sed -i "s/^EPAM_AUTH_SECRET_KEY=.*/EPAM_AUTH_SECRET_KEY=${AUTH_SECRET}/" "${ENV_FILE}"
  sed -i "s/^EPAM_ENCRYPTION_SECRET_KEY=.*/EPAM_ENCRYPTION_SECRET_KEY=${ENC_SECRET}/" "${ENV_FILE}"
  sed -i "s/^EPAM_AUTH_DEFAULT_PASSWORD=.*/EPAM_AUTH_DEFAULT_PASSWORD=${BOOTSTRAP_PASSWORD}/" "${ENV_FILE}"
  echo "[SETUP] Created random auth/encryption secrets."
  echo "[SETUP] Created one-time bootstrap admin password: ${BOOTSTRAP_PASSWORD}"
  echo "[SETUP] Change it after first login, then clear EPAM_AUTH_DEFAULT_PASSWORD or replace it with a hash."
  echo "[SETUP] Open ${ENV_FILE} and add HF_TOKEN before the first pyannote run."
  echo
fi

set -a
# shellcheck disable=SC1090
. "${ENV_FILE}"
set +a

if [[ "${VLLM_IMAGE:-}" == *":latest" ]]; then
  echo "[ERROR] VLLM_IMAGE must not use :latest for a server deployment."
  echo "        Pin an approved tag or digest in ${ENV_FILE}."
  exit 4
fi

if [[ "${OLLAMA_IMAGE:-}" == *":latest" ]]; then
  echo "[ERROR] OLLAMA_IMAGE must not use :latest for a server deployment."
  echo "        Pin an approved tag or digest in ${ENV_FILE}."
  exit 4
fi

if [ "${EPAM_FRONTEND_BIND:-127.0.0.1}" != "127.0.0.1" ] && [ -z "${EPAM_AUTH_DEFAULT_PASSWORD_HASH:-}" ]; then
  echo "[ERROR] Frontend is configured for non-local access, but no admin password hash is set."
  echo "        Set EPAM_AUTH_DEFAULT_PASSWORD_HASH before exposing VERITAS to the network."
  exit 5
fi

if [ -z "${EPAM_AUTH_DEFAULT_PASSWORD_HASH:-}" ] && [ "${EPAM_AUTH_DEFAULT_PASSWORD:-admin}" = "admin" ]; then
  echo "[SECURITY] Bootstrap admin password is still the built-in default."
  echo "[SECURITY] Keep EPAM_FRONTEND_BIND=127.0.0.1 and change it immediately."
  echo
fi

if [ "${EPAM_FRONTEND_BIND:-127.0.0.1}" = "127.0.0.1" ]; then
  echo "[SECURITY] Frontend is bound to localhost only. Use VPN/reverse proxy after IT review."
else
  echo "[SECURITY] Frontend bind is ${EPAM_FRONTEND_BIND}. Confirm firewall/VPN controls are in place."
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARNING] HF_TOKEN is empty. First pyannote model download may fail."
fi

LLM_PROVIDER="${EPAM_SUMMARIZATION_PROVIDER:-ollama}"
OLLAMA_MODEL="${EPAM_SUMMARIZATION_OLLAMA_MODEL:-gemma4:26b}"
if [ "${LLM_PROVIDER}" = "ollama" ] && [ "${OLLAMA_MODEL}" != "gemma4:26b" ]; then
  echo "[ERROR] Server baseline must start with Gemma 4 via Ollama."
  echo "        Current EPAM_SUMMARIZATION_OLLAMA_MODEL=${OLLAMA_MODEL}"
  echo "        Set EPAM_SUMMARIZATION_OLLAMA_MODEL=gemma4:26b for pilot deployment."
  echo "        Alternative models must be enabled deliberately after baseline validation."
  exit 6
fi

if [[ ",${COMPOSE_PROFILES:-}," == *",vllm,"* ]] && [ "${LLM_PROVIDER}" != "openai_compatible" ]; then
  echo "[ERROR] COMPOSE_PROFILES includes vllm, but EPAM_SUMMARIZATION_PROVIDER is ${LLM_PROVIDER}."
  echo "        Remove vllm from COMPOSE_PROFILES for the Gemma/Ollama baseline."
  exit 7
fi

TOTAL_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.62}"
BUDGET="${VERITAS_GPU_VRAM_BUDGET_GB:-30}"
VLLM_ALLOC_GB="$(awk -v mb="${TOTAL_MB}" -v u="${UTIL}" 'BEGIN { printf "%.1f", mb*u/1024 }')"

echo "[GPU] ${GPU_NAME}"
echo "[GPU] Total VRAM: $((TOTAL_MB / 1024)) GB"
echo "[GPU] VERITAS declared LLM budget: ${BUDGET} GB"
echo "[GPU] vLLM gpu_memory_utilization=${UTIL} => approx ${VLLM_ALLOC_GB} GB"
echo
echo "[START] Building and starting VERITAS..."
echo "[START] Compose file: ${COMPOSE_FILE}"
echo

cd "${REPO_ROOT}"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up --build -d

if [ "${LLM_PROVIDER}" = "ollama" ]; then
  echo
  echo "[OLLAMA] Baseline LLM: ${OLLAMA_MODEL}"
  echo "[OLLAMA] Waiting for Ollama service..."
  for _ in $(seq 1 90); do
    OLLAMA_STATUS="$(docker inspect -f '{{.State.Health.Status}}' veritas-ollama 2>/dev/null || true)"
    if [ "${OLLAMA_STATUS}" = "healthy" ]; then
      break
    fi
    sleep 2
  done
  OLLAMA_STATUS="$(docker inspect -f '{{.State.Health.Status}}' veritas-ollama 2>/dev/null || true)"
  if [ "${OLLAMA_STATUS}" != "healthy" ]; then
    echo "[ERROR] Ollama did not become healthy. Check logs:"
    echo "        docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs -f ollama"
    exit 8
  fi
  echo "[OLLAMA] Ensuring the model is available. The first pull can take a while."
  docker exec veritas-ollama ollama pull "${OLLAMA_MODEL}"
  docker exec veritas-ollama ollama show "${OLLAMA_MODEL}" >/dev/null
fi

echo
echo "[WAIT] Waiting for backend health..."
for _ in $(seq 1 90); do
  STATUS="$(docker inspect -f '{{.State.Health.Status}}' veritas-backend 2>/dev/null || true)"
  if [ "${STATUS}" = "healthy" ]; then
    echo "[OK] Backend is healthy."
    break
  fi
  sleep 2
done

STATUS="$(docker inspect -f '{{.State.Health.Status}}' veritas-backend 2>/dev/null || true)"
if [ "${STATUS}" != "healthy" ]; then
  echo "[WARNING] Containers started, but backend is not healthy yet."
  echo "          Check logs: docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs -f backend"
fi

SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -z "${SERVER_IP}" ]; then
  SERVER_IP="SERVER_IP"
fi

echo
echo "======================================================"
echo " VERITAS 1.0 is running"
echo
echo " Local URL:  http://localhost:${EPAM_FRONTEND_PORT:-5173}"
if [ "${EPAM_FRONTEND_BIND:-127.0.0.1}" = "127.0.0.1" ]; then
  echo " Network:    localhost-only. Publish through an IT-approved reverse proxy/VPN."
else
  echo " LAN URL:    http://${SERVER_IP}:${EPAM_FRONTEND_PORT:-5173}"
fi
echo " Login:      use the configured admin account; change bootstrap password before pilot use."
if [ -n "${BOOTSTRAP_PASSWORD}" ]; then
  echo " Bootstrap:  admin / ${BOOTSTRAP_PASSWORD}"
fi
echo
echo " Stop:       ${SCRIPT_DIR}/STOP_VERITAS_UBUNTU.sh"
echo " Logs:       docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs -f"
echo "======================================================"
echo
