#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.server"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.server.yml"

cd "${REPO_ROOT}"

if [ -f "${ENV_FILE}" ]; then
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" down
else
  docker compose -f "${COMPOSE_FILE}" down
fi

echo "VERITAS containers stopped."
