#!/usr/bin/env bash
# ============================================================

# 64 total runs: 4 methods × 2 tasks × 8 seeds
#
# Strategy:
#   Batch 1: peg-insert-side-v3 (32 runs)  ← priority
#   Batch 2: pick-place-v3      (32 runs)  ← auto-starts when batch 1 done
#
# Sessions named: ca_{method}_{task_short}_{seed}
#   e.g. ca_A_peg_0, ca_D_pp_7
# ============================================================

set -e

BASE="/home/deep/projects/rlp"
SCRIPT="$BASE/experiments/exp35_causal_ablation/train_ablation.py"
PYTHON="$BASE/env/bin/python3"
LOGBASE="$BASE/experiments/exp35_causal_ablation/logs"
THREADS=3   # OMP/MKL threads per process (32 × 3 = 96 of 112 cores)

METHODS=(A B C D)
SEEDS=(0 1 2 3 4 5 6 7)

mkdir -p "$LOGBASE"

# ── Helper: launch one run in a detached tmux session ──────
launch_run() {
    local method=$1
    local task=$2
    local task_short=$3
    local seed=$4

    local session="ca_${method}_${task_short}_${seed}"
    local logfile="$LOGBASE/${task}__method${method}__seed${seed}.log"

    # Skip if already complete
    local done_marker="$LOGBASE/${task}__method${method}__seed${seed}/first_success_step.txt"
    if [ -f "$done_marker" ]; then
        echo "[SKIP] $session — already done"
        return
    fi

    # Kill stale session if exists
    tmux kill-session -t "$session" 2>/dev/null || true

    tmux new-session -d -s "$session" \
        "cd $BASE && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        $PYTHON $SCRIPT \
            --method $method \
            --task $task \
            --seed $seed \
            --steps 1000000 \
            --logdir $LOGBASE \
        2>&1 | tee $logfile; \
        echo EXIT_CODE=\$? >> $logfile"

    echo "[LAUNCHED] $session"
}

# ── Helper: wait until all ca_{suffix} sessions finish ─────
wait_for_batch() {
    local pattern=$1   # e.g. "ca_.*_peg_"
    echo ""
    echo "========================================================"
    echo " Waiting for batch: $pattern"
    echo "========================================================"
    while true; do
        local running
        running=$(tmux ls 2>/dev/null | grep -cE "$pattern" || echo 0)
        if [ "$running" -eq 0 ]; then
            echo "[DONE] All sessions matching '$pattern' finished."
            break
        fi
        echo "  $(date '+%H:%M:%S') — $running sessions still running..."
        sleep 300   # poll every 5 minutes
    done
}

# ── BATCH 1: peg-insert-side-v3 ────────────────────────────
echo "========================================================"
echo " BATCH 1: peg-insert-side-v3 (32 runs)"
echo "========================================================"

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        launch_run "$method" "peg-insert-side-v3" "peg" "$seed"
    done
done

wait_for_batch "ca_.*_peg_"

echo ""
echo "[BATCH 1 COMPLETE] $(date)"
echo ""

# ── Checkpoint: quick result summary ───────────────────────
echo "=== QUICK RESULTS (peg-insert, last eval) ==="
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        local_dir="$LOGBASE/peg-insert-side-v3__method${method}__seed${seed}"
        fss="$local_dir/first_success_step.txt"
        if [ -f "$fss" ]; then
            echo "  method=$method seed=$seed first_success=$(cat $fss)"
        fi
    done
done

# ── BATCH 2: pick-place-v3 ─────────────────────────────────
echo ""
echo "========================================================"
echo " BATCH 2: pick-place-v3 (32 runs)"
echo "========================================================"

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        launch_run "$method" "pick-place-v3" "pp" "$seed"
    done
done

wait_for_batch "ca_.*_pp_"

echo ""
echo "[BATCH 2 COMPLETE — ALL 64 RUNS DONE] $(date)"
echo ""

# ── Final summary ───────────────────────────────────────────
echo "=== FINAL FIRST-SUCCESS SUMMARY ==="
for task_short in peg pp; do
    if [ "$task_short" = "peg" ]; then task="peg-insert-side-v3"; else task="pick-place-v3"; fi
    echo "--- $task ---"
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            fss="$LOGBASE/${task}__method${method}__seed${seed}/first_success_step.txt"
            val=$(cat "$fss" 2>/dev/null || echo "missing")
            echo "  $method s$seed: $val"
        done
    done
done

echo ""
echo "[ALL DONE] Results in $LOGBASE"
