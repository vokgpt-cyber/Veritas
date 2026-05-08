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
  sed -i "s/^EPAM_AUTH_SECRET_KEY=.*/EPAM_AUTH_SECRET_KEY=${AUTH_SECRET}/" "${ENV_FILE}"
  sed -i "s/^EPAM_ENCRYPTION_SECRET_KEY=.*/EPAM_ENCRYPTION_SECRET_KEY=${ENC_SECRET}/" "${ENV_FILE}"
  echo "[SETUP] Created random auth/encryption secrets."
  echo "[SETUP] Open ${ENV_FILE} and add HF_TOKEN before the first pyannote run."
  echo
fi

set -a
# shellcheck disable=SC1090
. "${ENV_FILE}"
set +a

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
echo " LAN URL:    http://${SERVER_IP}:${EPAM_FRONTEND_PORT:-5173}"
echo " Login:      admin / admin"
echo
echo " Stop:       ${SCRIPT_DIR}/STOP_VERITAS_UBUNTU.sh"
echo " Logs:       docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs -f"
echo "======================================================"
echo
