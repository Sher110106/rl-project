#!/usr/bin/env bash
# =============================================================
# test.sh - Automated showcase for the best trained SAC model
#
# Target: Ubuntu 22.04 (Docker: ubuntu:22.04)
# Usage:  bash test.sh
# =============================================================

set -euo pipefail

echo "=============================================="
echo " Trained SAC Model Showcase - test.sh         "
echo "=============================================="

timestamp() {
    date '+%H:%M:%S'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$1"
}

if ! command -v apt-get >/dev/null 2>&1; then
    printf 'Error: this script expects Ubuntu/Debian with apt-get.\n' >&2
    exit 1
fi

APT_CMD=(apt-get)
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        APT_CMD=(sudo apt-get)
    else
        printf 'Error: not running as root and sudo is not installed.\n' >&2
        exit 1
    fi
fi

export DEBIAN_FRONTEND=noninteractive

apt_update_with_retry() {
    local attempt max_attempts sleep_seconds
    max_attempts=5

    for attempt in $(seq 1 "${max_attempts}"); do
        log "Running apt-get update (attempt ${attempt}/${max_attempts})..."
        if "${APT_CMD[@]}" clean && rm -rf /var/lib/apt/lists/* && "${APT_CMD[@]}" update -o Acquire::Retries=3; then
            log "apt package index refreshed successfully."
            return 0
        fi

        if [ "${attempt}" -lt "${max_attempts}" ]; then
            sleep_seconds=$((attempt * 10))
            log "apt-get update failed, likely due to mirror sync or network issues. Retrying in ${sleep_seconds}s..."
            sleep "${sleep_seconds}"
        fi
    done

    printf 'Error: apt-get update failed after %s attempts.\n' "${max_attempts}" >&2
    printf 'This is usually a temporary Ubuntu mirror-sync issue inside the container.\n' >&2
    exit 1
}

log "[0/4] Installing system dependencies for evaluation..."
apt_update_with_retry
"${APT_CMD[@]}" install -y --no-install-recommends -o Acquire::Retries=3 \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6
log "      System dependencies installed."

log "[1/4] Preparing virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
else
    log "      Reusing existing venv/."
fi

# shellcheck disable=SC1091
source venv/bin/activate
python3 -m pip install --upgrade pip
log "      Virtual environment ready."

log "[2/4] Installing Python dependencies..."
python3 -m pip install -r requirements.txt imageio imageio-ffmpeg
log "      Python dependencies installed."

log "[3/4] Verifying trained checkpoints exist..."
if [ ! -d "results/suction_gripper" ]; then
    printf 'Error: results/suction_gripper/ was not found.\n' >&2
    exit 1
fi
if ! find results/suction_gripper -path "*/models/best_model/best_model.zip" | grep -q .; then
    printf 'Error: no trained best_model.zip checkpoint was found under results/suction_gripper/.\n' >&2
    exit 1
fi
log "      Trained checkpoints detected."

log "[4/4] Running the best trained SAC model..."
if [ -n "${DISPLAY:-}" ]; then
    log "      DISPLAY detected - launching live PyBullet playback."
    python3 test_trained_model.py --live
else
    log "      No DISPLAY detected - recording a headless MP4 instead."
    python3 test_trained_model.py --headless
fi

log "=============================================="
log " Showcase complete."
log " If headless mode was used, check:"
log "   results/test_runs/"
log "=============================================="
