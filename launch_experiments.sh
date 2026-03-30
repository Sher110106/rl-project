#!/usr/bin/env bash
# launch_experiments.sh — Launch all 4 experiments in parallel tmux sessions
# Run from: ~/projects/rlp/
# Usage: bash launch_experiments.sh

set -e
BASE="$HOME/projects/rlp"
ENV_PYTHON="$BASE/env/bin/python"

echo "=== Launching RL Pick-Place Experiments on Tyrone ==="
echo "Base: $BASE"
echo "Python: $ENV_PYTHON"
echo ""

# --- exp4: Dense + Barrier ---
tmux new-session -d -s exp4_barrier -c "$BASE/experiments/exp4_barrier"
tmux send-keys -t exp4_barrier \
  "source $BASE/env/bin/activate && $ENV_PYTHON train_barrier.py --timesteps 1000000 \
   > $BASE/logs/exp4_barrier.log 2>&1; echo 'EXP4 DONE'" Enter
echo "[OK] exp4_barrier started"

# --- exp5: Dense + Granger ---
tmux new-session -d -s exp5_granger -c "$BASE/experiments/exp5_granger"
tmux send-keys -t exp5_granger \
  "source $BASE/env/bin/activate && $ENV_PYTHON train_granger.py --timesteps 1000000 \
   > $BASE/logs/exp5_granger.log 2>&1; echo 'EXP5 DONE'" Enter
echo "[OK] exp5_granger started"

# --- exp6: Curriculum Fix ---
tmux new-session -d -s exp6_curriculum -c "$BASE/experiments/exp6_curriculum_fix"
tmux send-keys -t exp6_curriculum \
  "source $BASE/env/bin/activate && $ENV_PYTHON train_curriculum_fix.py --timesteps 1500000 \
   > $BASE/logs/exp6_curriculum.log 2>&1; echo 'EXP6 DONE'" Enter
echo "[OK] exp6_curriculum_fix started"

# --- exp7: Grand Unified ---
tmux new-session -d -s exp7_combined -c "$BASE/experiments/exp7_combined"
tmux send-keys -t exp7_combined \
  "source $BASE/env/bin/activate && $ENV_PYTHON train_combined.py --timesteps 2000000 \
   > $BASE/logs/exp7_combined.log 2>&1; echo 'EXP7 DONE'" Enter
echo "[OK] exp7_combined started"

echo ""
echo "=== All 4 experiments running ==="
echo ""
echo "Monitor logs:"
echo "  tail -f $BASE/logs/exp4_barrier.log"
echo "  tail -f $BASE/logs/exp5_granger.log"
echo "  tail -f $BASE/logs/exp6_curriculum.log"
echo "  tail -f $BASE/logs/exp7_combined.log"
echo ""
echo "Check sessions:  tmux ls"
echo "Attach to exp7:  tmux attach -t exp7_combined"
