#!/usr/bin/env bash
# launch_wave2.sh — Launch 8 Granger variant experiments (CPU-only)

# Usage: bash launch_wave2.sh

set -e
BASE="$HOME/projects/rlp"
ENV_PYTHON="$BASE/env/bin/python"

echo "=== Wave 2: Granger Variants (CPU-only) ==="
echo "Base: $BASE"
echo "Python: $ENV_PYTHON"
echo ""

# --- exp8: Granger Extended 2M ---
tmux new-session -d -s exp8_granger_2m -c "$BASE/experiments/exp8_granger_2m"
tmux send-keys -t exp8_granger_2m \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger.py --timesteps 2000000 \
   > $BASE/logs/exp8_granger_2m.log 2>&1; echo 'EXP8 DONE'" Enter
echo "[OK] exp8_granger_2m started (2M steps)"

# --- exp9: Granger seed=42 ---
tmux new-session -d -s exp9_seed42 -c "$BASE/experiments/exp9_granger_seed42"
tmux send-keys -t exp9_seed42 \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger.py --timesteps 1000000 \
   > $BASE/logs/exp9_seed42.log 2>&1; echo 'EXP9 DONE'" Enter
echo "[OK] exp9_granger_seed42 started (1M steps)"

# --- exp10: Granger seed=123 ---
tmux new-session -d -s exp10_seed123 -c "$BASE/experiments/exp10_granger_seed123"
tmux send-keys -t exp10_seed123 \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger.py --timesteps 1000000 \
   > $BASE/logs/exp10_seed123.log 2>&1; echo 'EXP10 DONE'" Enter
echo "[OK] exp10_granger_seed123 started (1M steps)"

# --- exp11: Granger seed=456 ---
tmux new-session -d -s exp11_seed456 -c "$BASE/experiments/exp11_granger_seed456"
tmux send-keys -t exp11_seed456 \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger.py --timesteps 1000000 \
   > $BASE/logs/exp11_seed456.log 2>&1; echo 'EXP11 DONE'" Enter
echo "[OK] exp11_granger_seed456 started (1M steps)"

# --- exp12: Granger fixed entropy ---
tmux new-session -d -s exp12_fixent -c "$BASE/experiments/exp12_granger_fixent"
tmux send-keys -t exp12_fixent \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger.py --timesteps 1000000 \
   > $BASE/logs/exp12_fixent.log 2>&1; echo 'EXP12 DONE'" Enter
echo "[OK] exp12_granger_fixent started (1M steps)"

# --- exp13: Granger CAUSAL_SCALE=0.2 ---
tmux new-session -d -s exp13_scale02 -c "$BASE/experiments/exp13_granger_scale02"
tmux send-keys -t exp13_scale02 \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger.py --timesteps 1000000 \
   > $BASE/logs/exp13_scale02.log 2>&1; echo 'EXP13 DONE'" Enter
echo "[OK] exp13_granger_scale02 started (1M steps)"

# --- exp14: Granger CAUSAL_SCALE=0.8 ---
tmux new-session -d -s exp14_scale08 -c "$BASE/experiments/exp14_granger_scale08"
tmux send-keys -t exp14_scale08 \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger.py --timesteps 1000000 \
   > $BASE/logs/exp14_scale08.log 2>&1; echo 'EXP14 DONE'" Enter
echo "[OK] exp14_granger_scale08 started (1M steps)"

# --- exp15: Granger + Barrier ---
tmux new-session -d -s exp15_barrier -c "$BASE/experiments/exp15_granger_barrier"
tmux send-keys -t exp15_barrier \
  "source $BASE/env/bin/activate && CUDA_VISIBLE_DEVICES='' $ENV_PYTHON train_granger_barrier.py --timesteps 1000000 \
   > $BASE/logs/exp15_barrier.log 2>&1; echo 'EXP15 DONE'" Enter
echo "[OK] exp15_granger_barrier started (1M steps)"

echo ""
echo "=== All 8 experiments running (CPU-only) ==="
echo ""
echo "Monitor logs:"
echo "  tail -f $BASE/logs/exp8_granger_2m.log"
echo "  tail -f $BASE/logs/exp9_seed42.log"
echo "  tail -f $BASE/logs/exp10_seed123.log"
echo "  tail -f $BASE/logs/exp11_seed456.log"
echo "  tail -f $BASE/logs/exp12_fixent.log"
echo "  tail -f $BASE/logs/exp13_scale02.log"
echo "  tail -f $BASE/logs/exp14_scale08.log"
echo "  tail -f $BASE/logs/exp15_barrier.log"
echo ""
echo "Check sessions:  tmux ls"
