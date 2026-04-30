#!/bin/bash
# fetch_diagnostics.sh

# Run: bash fetch_diagnostics.sh

REMOTE_USER=deep

REMOTE_DIR=~/projects/rlp/logs
LOCAL_DIR=results/suction_gripper

echo "Fetching episode diagnostics from ${REMOTE_HOST}..."

for seed in 0 1 2; do
    echo "  Seed ${seed}..."
    mkdir -p ${LOCAL_DIR}/seed${seed}/diagnostics
    scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/sac_suction_seed${seed}/diagnostics/episode_diagnostics.csv \
        ${LOCAL_DIR}/seed${seed}/diagnostics/
done

echo "Done. Files saved to:"
for seed in 0 1 2; do
    echo "  ${LOCAL_DIR}/seed${seed}/diagnostics/episode_diagnostics.csv"
done
