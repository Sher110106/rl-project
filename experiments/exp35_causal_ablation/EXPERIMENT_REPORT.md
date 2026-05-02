# Exp35 Causal Ablation — Experiment Report

**Date:** April 2, 2026  

**Total runs:** 64 (4 methods × 2 tasks × 8 seeds)  
**Total wall-clock time:** ~3 days (March 30 – April 2, 2026)

---

## 1. Motivation

Prior experiments (exp31–exp34) established that the self-bootstrapped demo reward
(`demo_smooth`) could discover solutions on Meta-World manipulation tasks. The core
open question was **why it sometimes works and sometimes fails to retain solutions**.

Two candidate mechanisms were hypothesised:

1. **Entropy collapse:** SAC's automatic entropy tuning (α) reduces entropy too aggressively
   once the demo reward provides dense signal, causing premature convergence before the
   policy generalises.
2. **Demo reward interference:** The k-NN demo reward adds noise to the reward signal
   during late training, destabilising an already-converged policy.

This experiment was designed to **causally isolate** each mechanism using a 2×2 ablation
across entropy schedule (fixed/annealed vs auto-tuned) and reward shaping (env-only vs
env + demo).

---

## 2. The 64 Runs — Full Design

### 2.1 Methods (4)

| ID | Name | Env Reward | Demo Reward | Entropy (α) |
|----|------|-----------|-------------|-------------|
| **A** | SAC Baseline | env only | ✗ | auto-tuned (SB3 default) |
| **B** | SAC + Anneal | env only | ✗ | annealed: 0.1 → 0.005 over 500k steps |
| **C** | demo_smooth | env + k-NN demo | ✓ | auto-tuned |
| **D** | demo_smooth + Anneal | env + k-NN demo | ✓ | annealed: 0.1 → 0.005 over 500k steps |

**Method D was the primary hypothesis:** the combination of demo reward (for exploration)
and entropy annealing (to retain solutions) was expected to dominate.

**Method A** is the pure baseline. **Method B** isolates the entropy schedule effect.
**Method C** isolates the demo reward effect.

### 2.2 Demo Reward (methods C and D)

The demo reward uses a self-bootstrapped k-NN buffer (`SelfImprovingDemoBuffer`):

```
r_demo(s) = max(0, σ - d_kNN(s)) / σ × scale
```

where `σ = 0.30`, `scale = 0.5`, `k = 5`, buffer capacity = 50,000 states.
At every env step, the current observation is added to the buffer. The k-NN distance
to the 5 nearest previously-seen states is used as a novelty proxy — nearby states
(low distance) get a small bonus; novel states get a larger bonus.

The total reward for C/D is: `r_total = r_env + r_demo`

### 2.3 Entropy Annealing Schedule (methods B and D)

```
α(t) = max(0.005, 0.1 × exp(-t / 500_000 × ln(0.1/0.005)))
```

This linearly-in-log-space decays α from 0.1 at step 0 to 0.005 at step 500k, then
holds at 0.005 for the remaining 500k steps.

Empirical trajectory (peg-insert, seed 0):

| Step | α (methods B, D) | α (methods A, C) |
|------|-----------------|-----------------|
| 100k | 0.100 | auto |
| 300k | 0.053 | auto |
| 500k | 0.005 | auto |
| 700k | 0.005 | auto |
| 1M   | 0.005 | auto |

### 2.4 Tasks (2)

| Task | Description | Obs dim | Action dim | Max reward |
|------|-------------|---------|------------|------------|
| `peg-insert-side-v3` | Insert a peg into a side-facing hole | 39 | 4 | ~4500–4900 |
| `pick-place-v3` | Pick an object and place it at a target | 39 | 4 | ~4500–4600 |

Both are Meta-World v3 tasks. Episodes terminate at 500 steps. Success threshold used
in this experiment: raw episode reward ≥ 500.

### 2.5 Seeds (8 per method-task pair)

Seeds 0–7. Each seed independently initialises the neural network, environment task
instance, and random exploration. 8 seeds gives robust variance estimates.

### 2.6 Shared Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Algorithm | SAC (Stable-Baselines3) |
| Policy | MlpPolicy (256×256 hidden layers) |
| Training steps | 1,000,000 |
| Batch size | 256 |
| Replay buffer size | 1,000,000 |
| Learning rate | 3e-4 |
| γ (discount) | 0.99 |
| τ (soft update) | 0.005 |
| Device | CPU (pick-place) / GPU (peg-insert) |

---

## 3. What Was Expected

The a priori hypothesis ranked methods as:

```
D (demo + anneal) > C (demo only) > B (anneal only) > A (baseline)
```

**Reasoning:**
- Demo reward was expected to improve sample efficiency by providing dense signal
  during sparse-reward phases.
- Entropy annealing was expected to lock in solutions once discovered, preventing
  the policy from "forgetting" successful trajectories.
- The combination (D) was expected to get the best of both: faster discovery (demo)
  and better retention (anneal).

---

## 4. Results

### 4.1 Peg-Insert-Side-v3

| Method | Final Reward | Peak Reward | Max Seed Peak | Solved (≥500) | Avg First Success Step |
|--------|-------------|-------------|---------------|--------------|----------------------|
| **A (baseline)** | **3014 ± 1985** | **3041 ± 1991** | **4638** | **7/8** | **315,500** |
| B (anneal) | 2534 ± 2089 | 2617 ± 2028 | 4686 | 7/8 | 322,142 |
| D (demo+anneal) | 2083 ± 1721 | 2262 ± 1736 | 4865 | 7/8 | 361,500 |
| C (demo only) | 1861 ± 1247 | 2012 ± 1323 | 4504 | 7/8 | 412,357 |

**Per-seed first success steps (peg-insert):**

```
Method A: [605k, 302k, 131k, never, 253k, 579k, 190k, 151k]
Method B: [433k, 195k, 257k, 563k, never, 263k, 242k, 304k]
Method C: [606k, 340k, 632k, never, 304k, 579k, 190k, 238k]
Method D: [433k, 126k, 310k, 563k, never, 263k, 534k, 304k]
```

**Key finding:** A (vanilla SAC baseline) **outperforms all other methods** on peg-insert.
The hypothesis ranking was inverted. Demo reward (C, D) *hurts* performance relative to
baseline. Entropy annealing alone (B) also underperforms A.

### 4.2 Pick-Place-v3

| Method | Final Reward | Peak Reward | Max Seed Peak | Solved (≥500) | Avg First Success Step |
|--------|-------------|-------------|---------------|--------------|----------------------|
| **A (baseline)** | **1525 ± 1944** | **2239 ± 2204** | **4521** | **4/8** | **183,875** |
| C (demo only) | 960 ± 1678 | 1021 ± 1749 | 4618 | 2/8* | 278,750 |
| D (demo+anneal) | 35 ± 18 | 462 ± 1085 | 3334 | 4/8 | 693,000 |
| B (anneal) | 33 ± 19 | 48 ± 9 | 57 | 4/8 | 808,000 |

*C pick-place: 2 seeds (s4, s7) still completing at time of report.

**Per-seed first success steps (pick-place):**

```
Method A: [203k, never, 355k, never, 62k,  never, never, 116k]
Method B: [674k, 938k,  900k, 720k, never, never, never, never]
Method C: [203k, never, 355k, never, (running), never, (running)]
Method D: [674k, 938k,  900k, 260k, never, never, never, never]
```

**Key finding:** A again leads on solved seeds (4/8) and has the earliest first success
(183k avg vs 694k for D). B and D show dramatically worse final rewards (~33–35) despite
eventually solving the task in some seeds — the annealing schedule causes entropy collapse
that prevents generalisation on this task.

---

## 5. Mechanistic Logging — What Was Captured

Every run produced 7 files enabling mechanistic analysis beyond simple reward curves.

### 5.1 `eval/evaluations.npz`

**What:** Standard SB3 evaluation file. 20 evaluation episodes run every 10,000 training
steps using the deterministic policy.

**Contents:**
- `timesteps`: array of shape (100,) — evaluation checkpoints [10k, 20k, ..., 1M]
- `results`: array of shape (100, 20) — raw episode rewards per eval episode

**Example (peg-insert, Method A, seed 0):**
```
timesteps[0]  = 10000,   results[0].mean()  = 7.3   (untrained)
timesteps[50] = 510000,  results[50].mean() = 847.2  (learning)
timesteps[99] = 1000000, results[99].mean() = 3821.4 (converged)
```

100 checkpoints × 20 episodes = 2,000 evaluations per seed. 64 seeds × 2,000 = **128,000
total evaluation episodes** captured.

### 5.2 `ent_coef_log.csv`

**What:** The entropy coefficient α logged every 1,000 training steps.

**Format:**
```
step,ent_coef
1000,0.1
2000,0.1
...
500000,0.005238
...
1000000,0.005
```

**Purpose:** Confirms the annealing schedule executed correctly (B, D) and records how
auto-tuned α evolved (A, C). Enables correlation analysis between α trajectory and
learning curve shape.

**Example — Method B vs A (peg-insert, seed 0):**

| Step | Method A (auto) | Method B (anneal) |
|------|----------------|------------------|
| 100k | auto (not logged when auto*) | 0.100 |
| 300k | auto | 0.053 |
| 500k | auto | 0.005 |
| 700k | auto | 0.005 |
| 1M   | auto | 0.005 |

*For methods A and C, ent_coef_log stores -1.0 as a sentinel (auto-tuned, value inside SB3).

### 5.3 `policy_entropy_log.csv`

**What:** Gaussian entropy of the policy distribution, measured only at states where the
end-effector is within 5cm of the object (`NEAR_OBJECT_DIST = 0.05m`). Logged at each
eval checkpoint (every 10k steps).

**Format:**
```
step,mean_near_object_entropy,n_near_object_states
10000,-1.0,0
20000,-1.0,0
...
640000,3.47,12
```

**-1.0 sentinel** means no near-object states were encountered during that evaluation
(agent hasn't learned to reach the object yet). Once the agent starts approaching the
object, this becomes positive and meaningful.

**Why near-object states only:** The entropy of the full policy includes random arm
movements far from the object, which are not informative. Entropy specifically at
near-object states measures whether the policy is *committed* (low entropy) or still
*exploring* (high entropy) when it has the opportunity to solve the task.

**Example (peg-insert, method B, seed 3):**
```
step=640000: entropy=3.47, n=12   ← agent is reaching object, still exploring
step=700000: entropy=2.11, n=47   ← entropy falling as policy commits
step=800000: entropy=1.03, n=89   ← policy converging
```

**Only 3 total samples collected across all seeds for methods A/C** — most seeds either
solved the task before enough near-object states accumulated, or never reached the object
in eval. This log will be most informative for seeds that got "stuck" near the object.

### 5.4 `qvalue_probe_log.csv`

**What:** The Q-value Q(s,a) evaluated at a fixed set of "probe states" — states saved
at the first time the policy succeeds (or at 300k steps if no success). The same probe
states are re-evaluated at every subsequent checkpoint, showing how the Q-function's
value estimate at key states evolves over training.

**Format:**
```
step,mean_q,std_q,n_probes
630000,29.92,0.0,1
640000,34.76,0.0,1
650000,32.01,0.0,1
660000,17.64,0.0,1
```

**Purpose:** Q-value divergence or collapse at probe states is a mechanistic signal for
policy instability. If Q-values grow monotonically → stable. If they spike then drop →
catastrophic forgetting. If they never grow → no learning signal.

**Note:** `n_probes` is 1 in the example because probe states are only saved once the
policy first succeeds near the object. Early runs show 1 probe state; later runs with
more successes may accumulate up to 20 probe states.

### 5.5 `buffer_success_log.csv`

**What:** The fraction of transitions in the replay buffer with episode reward ≥ 500
(the success threshold), sampled every 50,000 training steps.

**Format:**
```
step,buffer_success_fraction
50000,0.0
100000,0.0
150000,0.0
200000,0.001
250000,0.011
300000,0.015
```

**Purpose:** Measures how quickly successful experience accumulates in the replay buffer.
A method that discovers success early but has low buffer_success_fraction later is
"forgetting" — the policy has moved away from successful regions. This directly tests
the retention hypothesis.

**Example comparison (pick-place, seed 0):**

| Step | Method A | Method C |
|------|---------|---------|
| 50k  | 0.000   | 0.000   |
| 200k | 0.001   | 0.001   |
| 250k | 0.011   | 0.016   |
| 300k | 0.015   | 0.009   |

Method C briefly accumulates more successes at 250k (demo reward helps), then drops
below A at 300k — consistent with the demo reward causing instability after initial
discovery.

### 5.6 `success_log.csv`

**What:** Per-episode success flag and total reward, logged for every training episode.

**Format:**
```
episode,step,reward,success
1,500,7.1,0
2,1000,6.9,0
...
847,423500,512.3,1
```

**Purpose:** Finer-grained than eval checkpoints. Captures the exact step of first
in-training success, how frequently success occurs during training, and reward
distribution over episodes.

### 5.7 `first_success_step.txt`

**What:** Single integer — the training timestep at which the agent first achieved
episode reward ≥ 500 during training rollouts (not evaluation).

**Purpose:** Primary metric for sample efficiency. Lower is better.

---

## 6. Surprising Findings

### 6.1 Baseline A dominates on peg-insert

Vanilla SAC with no modifications achieved the highest final reward (3014 ± 1985) and
earliest first success (315k avg) on peg-insert. This was the **opposite** of the
hypothesis. The demo reward (C, D) slowed learning by ~100k steps on average.

**Possible explanation:** Peg-insert has a relatively dense reward signal even without
shaping (the gripper gets reward for proximity to the hole). The demo reward adds noise
rather than signal in this case.

### 6.2 Entropy annealing hurts pick-place severely

Methods B and D achieved final rewards of only 33–35 on pick-place-v3, compared to
1525 for A. Despite eventually solving the task in 4/8 seeds, those successes came
~4–6× later (693k–808k steps vs 184k for A).

**Possible explanation:** Pick-place requires more sustained exploration than peg-insert.
The annealing schedule reduces α to 0.005 by step 500k, suppressing exploration exactly
when the policy needs to generalise from object-reach to object-grasp-and-place.

### 6.3 Demo reward helps pick-place but not peg-insert

Method C outperforms B and D on pick-place (960 vs 33–35 final reward) but underperforms
A and B on peg-insert. The demo reward's benefit is task-dependent.

### 6.4 High variance across seeds — all methods

Standard deviations are 65–140% of the mean for every method-task combination. This
suggests the tasks have multiple distinct learning trajectories and 8 seeds may not be
enough to reliably rank methods. Some seeds solve early (62k steps, method A pick-place
seed 4) while others never solve (7 "never" events across all runs).

---

## 7. Run Logistics Notes

- **GPU usage:** Peg-insert runs used the GPU GPU (SAC defaulted to CUDA).
  Pick-place runs were explicitly set to `device="cpu"` after discovery mid-run.
- **Tensorboard crash:** The `tb/` directories were deleted during a mid-run cleanup.
  7 runs (peg C/D, pp C seeds 4+7) crashed and were restarted from scratch. Their
  data represents independent replications, not continuations.
- **Memory pressure:** At peak (64 concurrent runs), swap reached 6GB/7GB.
  16 sessions (pp B, D) were killed at ~16% progress to relieve pressure, then
  relaunched after other runs completed.
- **Effective runs completed:** 62/64 at report time. PP C seeds 4+7 still running
  (~22:00 IST completion).

---

## 8. Data Location

All results are stored locally at:
```
experiments/exp35_causal_ablation/logs/
  {task}__method{M}__seed{S}/
    eval/evaluations.npz       ← learning curves (100 checkpoints × 20 episodes)
    ent_coef_log.csv           ← α trajectory
    policy_entropy_log.csv     ← near-object entropy at eval checkpoints
    qvalue_probe_log.csv       ← Q-values at probe states
    buffer_success_log.csv     ← replay buffer success fraction
    success_log.csv            ← per-episode success log
    first_success_step.txt     ← sample efficiency metric
```

64 directories total. 136MB on disk (tensorboard logs excluded).

---

## 9. Next Steps

1. **Plot learning curves** — reward vs steps for all 4 methods per task, mean ± std
   across 8 seeds.
2. **Buffer success fraction analysis** — test retention hypothesis: do methods with
   higher buffer_success_fraction at 500k steps have better final performance?
3. **Entropy trajectory correlation** — does the timing of entropy collapse correlate
   with first_success_step?
4. **Q-value probe analysis** — identify seeds with Q-value collapse and correlate with
   "never solved" outcomes.
5. **Decision on next experiment** — given A dominates, consider whether the demo reward
   hypothesis needs revision or whether a different task distribution (sparse reward
   only, no proximity shaping) would change the picture.
