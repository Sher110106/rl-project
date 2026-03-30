# Self-Bootstrapped Dense Reward Shaping for Robotic Manipulation

A research project investigating reward retention in reinforcement learning for robotic manipulation. Built on [Meta-World v3](https://meta-world.github.io/) with Stable-Baselines3 SAC.

**Core finding:** The bottleneck in sparse-reward manipulation is not exploration — it's *retention*. Policies discover solutions but forget them because SAC's entropy bonus keeps pushing exploration even after success. Entropy annealing combined with smooth k-NN demo rewards converts transient discovery into stable retention.

---

## Results (5-seed, 1M steps, Meta-World v3)

### peg-insert-side-v3

| Method | Last-20 Mean ± Std | vs SAC |
|---|---|---|
| SAC (baseline) | 601 ± 528 | — |
| demo_smooth (k=5) | 305 ± 274 | −49% |
| **smooth+anneal** | **817 ± 539** | **+36%** |

### pick-place-v3

| Method | Last-20 Mean ± Std |
|---|---|
| SAC (baseline) | 68 ± 126 |
| demo_smooth (k=5) | 6 ± 2 |
| smooth+anneal | 18 ± 8 |

Pick-place remains largely unsolved for all methods (high variance, 1/5 seeds succeed by chance). Peg-insert is the primary benchmark.

---

## Method

### Self-Bootstrapped Demo Buffer

No human demonstrations. The agent builds its own demo buffer from successful transitions during training using a BallTree k-NN in 6D feature space (end-effector position + object position).

```
demo reward = max(0, σ − nn_dist) / σ × scale
```

Parameters: `k=5`, `σ=0.30`, `scale=0.5`

### Entropy Annealing

SAC auto-tunes `ent_coef` (α) to maximize entropy, which causes the policy to keep exploring even after discovering the solution. A linear schedule overrides this:

```
ent_coef: 0.1 → 0.005  over steps 100k – 500k
```

This suppresses exploration once the demo buffer has sufficient coverage, allowing the policy to commit to discovered solutions.

### Why demo_smooth alone doesn't help

Smooth reward (k=5 instead of k=1) makes the reward landscape less noisy but does not fix the forgetting problem. Without entropy annealing, the agent still drifts away from good regions. The two components are complementary.

---

## Repository Structure

```
experiments/
├── exp1_her/                    # HER baseline (PyBullet)
├── exp2_dense/ – exp7_combined/ # Dense shaping iterations (PyBullet)
├── exp8/ – exp15/               # Granger-causal reward variants (PyBullet)
├── exp16/ – exp19/              # OT demo reward (PyBullet)
├── exp20_casid_metaworld/       # CASID wrapper + training scripts (Meta-World)
│   ├── casid_env.py             # CASIDWrapper, SelfImprovingDemoBuffer
│   ├── train_casid.py           # Ablation runner (full/no_filter/demo_only/demo_smooth)
│   ├── train_smooth_anneal.py   # demo_smooth + entropy annealing (main method)
│   └── train_replay_bias.py     # SIL-lite replay bias (explored, not used)
└── exp21_baseline_metaworld/    # Vanilla SAC baseline

results/
├── final_runs/                  # 5-seed final results (evaluations.npz only)
│   ├── final_sac/               # SAC seed 4
│   ├── final_demo_smooth/       # demo_smooth seeds 1–4
│   ├── final_anneal/            # smooth+anneal all seeds
│   ├── phase2_baseline/         # SAC seeds 1–3
│   ├── phase2_demo_smooth/      # demo_smooth seed 0
│   └── phase2_smooth_anneal/    # smooth+anneal peg seeds 0–2
├── experiments/                 # Earlier PyBullet + Meta-World wave results
├── final_summary.json           # Machine-readable per-seed data
├── collect_final_results.py     # Parses all npz files → paper table
└── RESULTS.md                   # Full experiment log across all 34+ runs

NEURIPS_PLAN.md                  # Venue and submission planning
PHASE2_PLAN.md                   # Phase 2 diagnosis and experiment design
RESEARCH_AND_PLAN.md             # Literature review and research direction
```

---

## Reproducing the Main Result

### Dependencies

```bash
pip install stable-baselines3 metaworld gymnasium scikit-learn numpy
```

### Run smooth+anneal (main method)

```bash
cd experiments/exp20_casid_metaworld
python train_smooth_anneal.py \
    --task peg-insert-side-v3 \
    --seed 0 \
    --steps 1000000 \
    --logdir ../../results/my_run
```

### Run SAC baseline

```bash
cd experiments/exp21_baseline_metaworld
python train_baseline.py \
    --task peg-insert-side-v3 \
    --seed 0 \
    --steps 1000000 \
    --logdir ../../results/my_run
```

### Parse results

```bash
python results/collect_final_results.py
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | SAC (SB3) |
| Steps | 1,000,000 |
| Eval frequency | 10,000 steps |
| Eval episodes | 10 |
| Demo buffer max size | 50,000 |
| k (NN) | 5 |
| σ (bandwidth) | 0.30 |
| scale | 0.5 |
| ent_start | 0.1 |
| ent_end | 0.005 |
| anneal_start | 100,000 |
| anneal_end | 500,000 |

---

## Experiment History

34+ experiments across two environments:

- **PyBullet (exp1–19):** Custom Franka Panda pick-and-place. Iterated through HER, dense shaping, curriculum, barrier functions, Granger-causal reward, OT demo reward.
- **Meta-World v3 (exp20–34+):** Ported to standard benchmark. Ablated CASID modes (full, no_filter, causal_only, demo_only). Identified retention as the core bottleneck. Implemented and validated smooth+anneal.

Full results in [`results/RESULTS.md`](results/RESULTS.md).
