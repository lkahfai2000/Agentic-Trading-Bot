#!/usr/bin/env bash
# run_lab.sh — Continuous orchestration for the Agentic Trading Bot.
#
# Runs bridge.py in the background and meta_loop.py in a loop.
# On hot-swap (exit 2): restarts bridge to load new strategy code.
# On no-swap  (exit 0): sleeps until the next 15m boundary, then reruns.
# On error    (exit 1): sends Telegram alert, halts for manual intervention.
#
# Usage:
#   chmod +x run_lab.sh
#   ./run_lab.sh              # production
#   DRY_RUN=true ./run_lab.sh # bridge runs with --dry-run
#
# macOS Launch Daemon:
#   cp com.agentic-trading-bot.run-lab.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.agentic-trading-bot.run-lab.plist

set -uo pipefail

# ── Constants ────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="${SCRIPT_DIR}/logs"
readonly WRAPPER_LOG="${LOG_DIR}/wrapper.log"
readonly META_LOOP_SLEEP=900    # seconds between meta_loop runs on exit 0

BRIDGE_PID=""

# ── Logging ──────────────────────────────────────────────────────────
log() {
    local ts
    ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "${ts}  [run_lab]  $*" | tee -a "$WRAPPER_LOG"
}

# ── Environment (.env sourcing) ──────────────────────────────────────
load_env() {
    local env_file="${SCRIPT_DIR}/.env"
    if [[ -f "$env_file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip comments and blank lines
            [[ -z "$line" || "$line" == \#* ]] && continue
            export "$line"
        done < "$env_file"
        log "Loaded environment from .env"
    else
        log "WARNING: No .env file found at ${env_file}"
    fi
}

# ── Python discovery ─────────────────────────────────────────────────
find_python() {
    if [[ -x "${SCRIPT_DIR}/venv/bin/python" ]]; then
        echo "${SCRIPT_DIR}/venv/bin/python"
    elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
        echo "${SCRIPT_DIR}/.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        echo ""
    fi
}

# ── Telegram alert (calls alerts.py library via python -c) ───────────
send_alert() {
    local reason="$1"
    # Pass reason via env var to avoid shell-injection in Python string
    ALERT_REASON="$reason" "$PYTHON" -c "
import os, time
from alerts import TelegramAlerter
a = TelegramAlerter()
a.system_pause(os.environ['ALERT_REASON'])
time.sleep(2)  # let background sender thread flush
" 2>/dev/null || log "WARNING: Failed to send Telegram alert"
}

# ── Bridge lifecycle ─────────────────────────────────────────────────
start_bridge() {
    log "Starting bridge.py ..."
    local bridge_args=(--log-dir "$LOG_DIR")
    if [[ "${DRY_RUN:-}" == "true" ]]; then
        bridge_args+=(--dry-run)
        log "  (dry-run mode)"
    fi
    "$PYTHON" "${SCRIPT_DIR}/bridge.py" "${bridge_args[@]}" >>"$WRAPPER_LOG" 2>&1 &
    BRIDGE_PID=$!
    log "bridge.py started  PID=${BRIDGE_PID}"
}

bridge_is_alive() {
    [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null
}

kill_bridge() {
    if bridge_is_alive; then
        log "Sending SIGTERM to bridge.py (PID ${BRIDGE_PID}) ..."
        kill "$BRIDGE_PID" 2>/dev/null
        # Wait up to 30s for graceful shutdown (bridge finally block runs audit.py)
        local i=0
        while kill -0 "$BRIDGE_PID" 2>/dev/null && (( i < 30 )); do
            sleep 1
            (( i++ ))
        done
        if kill -0 "$BRIDGE_PID" 2>/dev/null; then
            log "Bridge did not exit gracefully — sending SIGKILL"
            kill -9 "$BRIDGE_PID" 2>/dev/null
        fi
        wait "$BRIDGE_PID" 2>/dev/null || true
        log "bridge.py stopped"
    fi
    BRIDGE_PID=""
}

ensure_bridge() {
    if ! bridge_is_alive; then
        if [[ -n "$BRIDGE_PID" ]]; then
            # Bridge was running but exited on its own (e.g. System Pause)
            wait "$BRIDGE_PID" 2>/dev/null || true
            log "WARNING: bridge.py exited on its own — restarting"
        fi
        start_bridge
    fi
}

# ── Sleep with periodic bridge health checks ─────────────────────────
sleep_with_bridge_check() {
    local total=$1
    local elapsed=0
    while (( elapsed < total )); do
        sleep 60
        (( elapsed += 60 ))
        if ! bridge_is_alive; then
            wait "$BRIDGE_PID" 2>/dev/null || true
            log "WARNING: bridge.py died during sleep — will restart on next iteration"
            BRIDGE_PID=""
            break
        fi
    done
}

# ── Compute seconds until next 15-minute boundary ────────────────────
seconds_to_next_quarter() {
    local now_min now_sec min_in_quarter wait_s
    now_min=$(date -u +"%M")
    now_sec=$(date -u +"%S")
    # Strip leading zeros for arithmetic
    now_min=$((10#$now_min))
    now_sec=$((10#$now_sec))
    min_in_quarter=$(( now_min % 15 ))
    wait_s=$(( (15 - min_in_quarter) * 60 - now_sec ))
    # If we're exactly on a 15m boundary, wait a full 15 minutes
    if (( wait_s <= 0 )); then
        wait_s=900
    fi
    echo "$wait_s"
}

# ── Signal handling ──────────────────────────────────────────────────
cleanup() {
    log "Received shutdown signal — cleaning up ..."
    kill_bridge
    log "=== run_lab.sh exiting ==="
    exit 0
}
trap cleanup SIGTERM SIGINT

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

cd "$SCRIPT_DIR"
mkdir -p "$LOG_DIR"

load_env

PYTHON="$(find_python)"
if [[ -z "$PYTHON" ]]; then
    log "FATAL: No Python interpreter found (checked venv/bin/python, python3, python)"
    exit 1
fi

log "========================================"
log "  run_lab.sh starting"
log "  Python:  ${PYTHON}"
log "  Workdir: ${SCRIPT_DIR}"
log "  DryRun:  ${DRY_RUN:-false}"
log "========================================"

while true; do
    # 1. Ensure bridge is running
    ensure_bridge

    # 2. Run meta_loop (foreground, blocking)
    log "Running meta_loop.py --enable-llm ..."
    "$PYTHON" "${SCRIPT_DIR}/meta_loop.py" --enable-llm --log-dir "$LOG_DIR" >>"$WRAPPER_LOG" 2>&1
    META_EXIT=$?
    log "meta_loop.py exited with code ${META_EXIT}"

    # 3. Handle exit code
    case $META_EXIT in
        2)
            # Hot-swap: strategy file rewritten — must restart bridge
            log "HOT-SWAP detected (exit 2). Restarting bridge to load new strategy ..."
            kill_bridge
            sleep 5   # brief settle before restart
            # Cooldown: wait until next 15m boundary before re-running meta_loop.
            # Prevents rapid re-swaps if convergence logic has a bug.
            WAIT=$(seconds_to_next_quarter)
            log "Post-swap cooldown: sleeping ${WAIT}s until next 15m boundary ..."
            start_bridge
            sleep_with_bridge_check "$WAIT"
            ;;
        0)
            # No swap — sleep until next 15m boundary
            WAIT=$(seconds_to_next_quarter)
            log "No swap (exit 0). Sleeping ${WAIT}s until next 15m boundary ..."
            sleep_with_bridge_check "$WAIT"
            ;;
        *)
            # Error (exit 1 or unexpected) — alert and halt
            log "CRITICAL: meta_loop.py exited with code ${META_EXIT}"
            send_alert "run_lab.sh: meta_loop.py crashed (exit code ${META_EXIT}). Wrapper halting for manual intervention."
            kill_bridge
            log "=== run_lab.sh halting (manual intervention required) ==="
            exit 1
            ;;
    esac
done
