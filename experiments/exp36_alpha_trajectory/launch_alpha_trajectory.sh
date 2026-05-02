#!/usr/bin/env bash
# ============================================================
# exp36 launcher — α trajectory runs
#
# Runs Method A (SAC baseline) with full α logging on 5 tasks × 8 seeds.
#
# Batch 1: peg-insert-side-v3  (8 seeds)   ← already have exp35, re-run with α log
# Batch 2: pick-place-v3       (8 seeds)   ← already have exp35, re-run with α log
# Batch 3: door-open-v3        (8 seeds)   ← new task
# Batch 4: drawer-close-v3     (8 seeds)   ← new task
# Batch 5: window-open-v3      (8 seeds)   ← new task
#
# Session naming: e36_{task_short}_{seed}
#   e.g. e36_peg_0, e36_door_3
# ============================================================

set -e

BASE="/home/deep/projects/rlp"
SCRIPT="$BASE/experiments/exp36_alpha_trajectory/train_alpha_trajectory.py"
PYTHON="$BASE/env/bin/python3"
LOGBASE="$BASE/experiments/exp36_alpha_trajectory/logs"
THREADS=3
STEPS=1000000

mkdir -p "$LOGBASE"

declare -A TASK_SHORTS=(
    ["peg-insert-side-v3"]="peg"
    ["pick-place-v3"]="pp"
    ["door-open-v3"]="door"
    ["drawer-close-v3"]="drawer"
    ["window-open-v3"]="window"
)

TASKS=(
    "peg-insert-side-v3"
    "pick-place-v3"
    "door-open-v3"
    "drawer-close-v3"
    "window-open-v3"
)

SEEDS=(0 1 2 3 4 5 6 7)

launch_run() {
    local task=$1
    local seed=$2
    local tshort="${TASK_SHORTS[$task]}"
    local session="e36_${tshort}_${seed}"
    local logfile="$LOGBASE/${task}__seed${seed}.log"

    # Skip if already complete
    if [ -f "$LOGBASE/${task}__seed${seed}/first_success_step.txt" ] && \
       [ -f "$LOGBASE/${task}__seed${seed}/alpha_trajectory.csv" ]; then
        lines=$(wc -l < "$LOGBASE/${task}__seed${seed}/alpha_trajectory.csv")
        if [ "$lines" -gt 900 ]; then
            echo "[SKIP] $session — already complete"
            return
        fi
    fi

    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" \
        "cd $BASE && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        $PYTHON $SCRIPT \
            --task $task \
            --seed $seed \
            --steps $STEPS \
            --logdir $LOGBASE \
        2>&1 | tee $logfile; echo EXIT_CODE=\$? >> $logfile"
    echo "[LAUNCHED] $session"
}

echo "========================================================"
echo " exp36: α trajectory — 5 tasks × 8 seeds = 40 runs"
echo "========================================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- $task ---"
    for seed in "${SEEDS[@]}"; do
        launch_run "$task" "$seed"
    done
done

echo ""
echo "========================================================"
echo " All 40 sessions launched."
echo " Monitor with: tmux ls | grep e36_"
echo " Progress:     tail -f $LOGBASE/{task}__seed{N}.log"
echo "========================================================"
