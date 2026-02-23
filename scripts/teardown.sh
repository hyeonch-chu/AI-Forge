#!/usr/bin/env bash
# teardown.sh — Stop and remove the entire AI-Forge Docker Compose stack.
#
# Removes all containers, networks, and named volumes created by docker compose.
# Data volumes (postgres_data/, minio_data/) on the host filesystem are NOT
# deleted — remove them manually if a full clean slate is needed.
#
# Usage:
#   bash scripts/teardown.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

log() { echo "[teardown] $*"; }

log "Tearing down AI-Forge stack from: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

# Stop containers, remove networks and named volumes (grafana_data, loki_data).
# The --remove-orphans flag also cleans up containers from previous compose files.
docker compose down --volumes --remove-orphans

log "Stack removed successfully."
log "Note: host-mounted data directories (postgres_data/, minio_data/) were NOT deleted."
