#!/usr/bin/env bash
# =============================================================
# run.sh - Automated end-to-end pipeline
# Suction Gripper Pick-and-Place RL (SAC, PyBullet)
#
# Target: Ubuntu 22.04 (Docker: ubuntu:22.04)
# Usage:  bash run.sh
# =============================================================

set -euo pipefail

echo "=============================================="
echo " Suction Gripper Pick-and-Place RL - run.sh  "
echo "=============================================="

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
timestamp() {
    date '+%H:%M:%S'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$1"
}

section_start() {
    STEP_START_TIME=$(date +%s)
    log "$1"
}

section_done() {
    local now elapsed
    now=$(date +%s)
    elapsed=$((now - STEP_START_TIME))
    log "$1 (${elapsed}s)"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Error: required command not found: %s\n' "$1" >&2
        exit 1
    fi
}

pip_name_from_requirement() {
    local requirement="$1"
    requirement="${requirement%%#*}"
    requirement="${requirement%%[*}"
    requirement="${requirement%%>=*}"
    requirement="${requirement%%<=*}"
    requirement="${requirement%%==*}"
    requirement="${requirement%%~=*}"
    requirement="${requirement%%!=*}"
    requirement="${requirement%%>*}"
    requirement="${requirement%%<*}"
    requirement="${requirement%%;*}"
    printf '%s' "$requirement"
}

SYSTEM_PACKAGES=(
    python3
    python3-pip
    python3-venv
    python3-dev
    build-essential
    libgl1
    libglib2.0-0
    libsm6
    libxrender1
    libxext6
)

# ----------------------------------------------------------
# 0. System dependencies (Ubuntu 22.04 / Debian-based)
#    Runs as root in Docker; falls back to sudo if available.
# ----------------------------------------------------------
section_start "[0/4] Checking and installing system dependencies..."

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
log "Refreshing apt package index..."
"${APT_CMD[@]}" update -o Acquire::Retries=3

missing_system_packages=()
already_installed_system_packages=()
for pkg in "${SYSTEM_PACKAGES[@]}"; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
        already_installed_system_packages+=("$pkg")
    else
        missing_system_packages+=("$pkg")
    fi
done

if [ "${#already_installed_system_packages[@]}" -gt 0 ]; then
    log "Already installed system packages:"
    for pkg in "${already_installed_system_packages[@]}"; do
        log "  - ${pkg}"
    done
fi

if [ "${#missing_system_packages[@]}" -eq 0 ]; then
    log "All required system packages are already installed."
else
    log "Installing missing system packages:"
    for pkg in "${missing_system_packages[@]}"; do
        log "  - ${pkg}"
    done
    "${APT_CMD[@]}" install -y --no-install-recommends -o Acquire::Retries=3 "${missing_system_packages[@]}"
fi

section_done "System dependency step complete"

require_cmd python3

# ----------------------------------------------------------
# 1. Virtual environment
# ----------------------------------------------------------
section_start "[1/4] Preparing virtual environment..."

if [ ! -d "venv" ]; then
    log "Creating venv/ ..."
    python3 -m venv venv
else
    log "Reusing existing venv/ ..."
fi

# shellcheck disable=SC1091
source venv/bin/activate
log "Upgrading pip inside the virtual environment..."
python3 -m pip install --upgrade pip

section_done "Virtual environment ready"

# ----------------------------------------------------------
# 2. Python dependencies
# ----------------------------------------------------------
section_start "[2/4] Checking and installing Python dependencies..."

python3 -m pip --version

while IFS= read -r requirement || [ -n "$requirement" ]; do
    requirement="${requirement#"${requirement%%[![:space:]]*}"}"
    requirement="${requirement%"${requirement##*[![:space:]]}"}"

    if [ -z "$requirement" ] || [[ "$requirement" == \#* ]]; then
        continue
    fi

    package_name=$(pip_name_from_requirement "$requirement")
    if python3 -m pip show "$package_name" >/dev/null 2>&1; then
        log "Installing/updating ${requirement} (currently present in venv)..."
    else
        log "Installing ${requirement} ..."
    fi
    python3 -m pip install "$requirement"
done < requirements.txt

section_done "Python dependencies step complete"

# ----------------------------------------------------------
# 3. Smoke-test the environment
# ----------------------------------------------------------
section_start "[3/4] Verifying environment loads correctly..."
python3 - <<'PYEOF'
from pick_place_env_suction import PickPlaceSuctionEnv

env = PickPlaceSuctionEnv()
obs, _ = env.reset()
assert obs.shape == (24,), f"Expected obs shape (24,), got {obs.shape}"
assert env.action_space.shape == (4,), f"Expected action shape (4,), got {env.action_space.shape}"
env.close()
print("Environment check passed: obs=(24,)  action=(4,)")
PYEOF
section_done "Smoke test complete"

# ----------------------------------------------------------
# 4. Train - 3 seeds x 1,200,000 steps
#    Outputs -> logs/sac_suction_seed{0,1,2}/
# ----------------------------------------------------------
section_start "[4/4] Training SAC agent - 3 seeds x 1,200,000 steps"
log "Estimated runtime: ~8.5 hrs/seed on CPU, ~2 hrs/seed on GPU"

for SEED in 0 1 2; do
    seed_start=$(date +%s)
    log "Starting seed ${SEED} ..."
    python3 train_suction.py --seed "${SEED}" --timesteps 1200000
    seed_end=$(date +%s)
    log "Seed ${SEED} complete ($((seed_end - seed_start))s)"
done

section_done "Training step complete"

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
log "=============================================="
log "Training complete."
log "Results saved to:"
log "  logs/sac_suction_seed0/"
log "  logs/sac_suction_seed1/"
log "  logs/sac_suction_seed2/"
log ""
log "Each seed directory contains:"
log "  eval/evaluations.npz             eval checkpoints"
log "  models/best_model/best_model.zip best checkpoint"
log "  models/final_model.zip           final model"
log "  models/checkpoints/              periodic saves"
log "  tb/                              TensorBoard logs"
log "  diagnostics/episode_diagnostics.csv"
log "=============================================="
