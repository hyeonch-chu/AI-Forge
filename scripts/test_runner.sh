#!/usr/bin/env bash
# test_runner.sh — AI-Forge test orchestrator
#
# Starts the full Docker Compose stack, waits for services to be healthy,
# runs pytest inside each service container, then tears the stack down.
#
# Usage:
#   bash scripts/test_runner.sh            # run all suites
#   bash scripts/test_runner.sh inference  # run only inference tests
#   bash scripts/test_runner.sh train      # run only train tests
#
# Exit code mirrors pytest: 0 = all passed, non-zero = failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yaml}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"   # seconds to wait for services to be ready
PYTEST_ARGS="${PYTEST_ARGS:--v -q}"  # extra args forwarded to pytest

SUITE="${1:-all}"                    # all | inference | train

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${GREEN}[test_runner]${NC} $*"; }
warn() { echo -e "${YELLOW}[test_runner]${NC} $*"; }
err()  { echo -e "${RED}[test_runner]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_for_container() {
    local container="$1"
    local seconds=0
    log "Waiting for container '${container}' to be running..."
    until docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -q true; do
        sleep 2
        seconds=$((seconds + 2))
        if [[ ${seconds} -ge ${WAIT_TIMEOUT} ]]; then
            err "Timed out waiting for '${container}' after ${WAIT_TIMEOUT}s"
            return 1
        fi
    done
    log "Container '${container}' is running."
}

run_suite() {
    local container="$1"
    local label="$2"
    log "Running ${label} tests in container '${container}'..."
    if docker exec "${container}" pytest tests/ ${PYTEST_ARGS}; then
        log "${label} tests: ${GREEN}PASSED${NC}"
    else
        err "${label} tests: FAILED"
        FAILED_SUITES+=("${label}")
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FAILED_SUITES=()

log "Building and starting Docker Compose stack..."
docker compose -f "${COMPOSE_FILE}" up -d --build

# Wait for service containers
case "${SUITE}" in
    inference) wait_for_container inference ;;
    train)     wait_for_container trainer ;;
    all)
        wait_for_container inference
        wait_for_container trainer
        ;;
    *)
        err "Unknown suite '${SUITE}'. Use: all | inference | train"
        exit 1
        ;;
esac

# Run selected suites
case "${SUITE}" in
    inference) run_suite inference "inference_service" ;;
    train)     run_suite trainer   "train_service"     ;;
    all)
        run_suite inference "inference_service"
        run_suite trainer   "train_service"
        ;;
esac

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
if [[ ${#FAILED_SUITES[@]} -eq 0 ]]; then
    log "All test suites ${GREEN}PASSED${NC}."
    exit 0
else
    err "Failed suites: ${FAILED_SUITES[*]}"
    exit 1
fi
