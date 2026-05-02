#!/usr/bin/env bash
# =============================================================
# run.sh — Automated end-to-end pipeline
# Suction Gripper Pick-and-Place RL (SAC, PyBullet)
#
# Target: Ubuntu 22.04 (docker: ubuntu:22.04)
# Usage:  bash run.sh
# =============================================================

set -e

echo "=============================================="
echo " Suction Gripper Pick-and-Place RL — run.sh  "
echo "=============================================="

# ----------------------------------------------------------
# 0. System dependencies (Ubuntu 22.04)
#    Runs as root in docker; use sudo if not root.
# ----------------------------------------------------------
echo "[0/4] Installing system dependencies..."

APT_CMD="apt-get"
if [ "$(id -u)" -ne 0 ]; then
    APT_CMD="sudo apt-get"
fi

$APT_CMD update -qq
$APT_CMD install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

echo "      System dependencies installed."

# ----------------------------------------------------------
# 1. Virtual environment
# ----------------------------------------------------------
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
echo "      Virtual environment ready."

# ----------------------------------------------------------
# 2. Python dependencies
# ----------------------------------------------------------
echo "[2/4] Installing Python dependencies..."
pip install --quiet -r requirements.txt
echo "      Dependencies installed."

# ----------------------------------------------------------
# 3. Smoke-test the environment
# ----------------------------------------------------------
echo "[3/4] Verifying environment loads correctly..."
python3 - <<'PYEOF'
from pick_place_env_suction import PickPlaceSuctionEnv
env = PickPlaceSuctionEnv()
obs, _ = env.reset()
assert obs.shape == (24,), f"Expected obs shape (24,), got {obs.shape}"
assert env.action_space.shape == (4,), f"Expected action shape (4,), got {env.action_space.shape}"
env.close()
print("      Environment check passed: obs=(24,)  action=(4,)")
PYEOF

# ----------------------------------------------------------
# 4. Train — 3 seeds x 1,200,000 steps
#    Outputs → logs/sac_suction_seed{0,1,2}/
# ----------------------------------------------------------
echo "[4/4] Training SAC agent — 3 seeds x 1,200,000 steps"
echo "      (~8.5 hrs/seed on CPU; ~2 hrs/seed on GPU)"
echo ""

for SEED in 0 1 2; do
    echo "  --- Seed ${SEED} ---"
    python3 train_suction.py --seed ${SEED} --timesteps 1200000
    echo "  Seed ${SEED} complete."
    echo ""
done

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo "=============================================="
echo " Training complete."
echo " Results saved to:"
echo "   logs/sac_suction_seed0/"
echo "   logs/sac_suction_seed1/"
echo "   logs/sac_suction_seed2/"
echo ""
echo " Each seed directory contains:"
echo "   eval/evaluations.npz             eval checkpoints"
echo "   models/best_model/best_model.zip best checkpoint"
echo "   models/final_model.zip           final model"
echo "   models/checkpoints/              periodic saves"
echo "   tb/                              TensorBoard logs"
echo "   diagnostics/episode_diagnostics.csv"
echo "=============================================="
