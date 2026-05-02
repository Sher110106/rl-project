# Reinforcement Learning for Robotic Pick-and-Place
### Suction Gripper Control with SAC + Alpha Bifurcation Analysis

A full reinforcement learning pipeline for robotic manipulation. A **Kuka IIWA arm with a suction gripper** learns to pick up a cube and place it in a tray using **Soft Actor-Critic (SAC)** in **PyBullet**. The project also includes a mechanistic analysis of SAC's entropy coefficient (α) as a diagnostic tool for predicting learning success before training converges.

---

## Contributors

| Name | Role |
|---|---|
| **Sher Partap Singh** | Environment design, SAC training pipeline, results analysis |
| **Mannan Sharma** | Dependencies, usage documentation, baseline testing |
| **Mudasir Rasheed** | Exp35 causal ablation study — training & analysis |
| **Akshita Shukla** | Exp36 alpha trajectory experiment — implementation |
| **Aryan Chopra** | Exp36 alpha trajectory experiment — analysis & plots |

---

## Quick Start

```bash
bash run.sh
```

This will create a virtual environment, install all dependencies, verify the environment, and train the SAC agent for 3 seeds × 1,200,000 steps. See [run.sh](run.sh) for full details.

---

## Project Overview

### The Task

A **Kuka IIWA robot arm** in PyBullet must learn to:
1. **Reach** a randomly placed cube on a table
2. **Activate suction** when the end-effector is within 8 cm of the cube
3. **Lift and carry** the cube to a tray
4. **Place** the cube within 5 cm of the tray centre

### Key Design Decision: IK-Resolved End-Effector Control

The core insight that made this task solvable was switching from direct joint control to **end-effector delta control resolved via Inverse Kinematics (IK)**:

| | Direct Joint Control | **EE-Delta + IK (This Work)** |
|---|---|---|
| **Action space** | 8D: `[j1..j7, suction]` | **4D: `[dx, dy, dz, suction]`** |
| **What the agent learns** | IK mapping + manipulation simultaneously | **Just where to move** |
| **Success rate** | **0%** | **16.6%** |

By letting PyBullet handle the IK, the agent only needs to reason about *where to move the hand* rather than solving inverse kinematics implicitly — a much simpler credit assignment problem.

---

## Environment Specification

### State Space — 24 dimensions

```
[joint_angles(7), joint_velocities(7), ee_pos(3), object_pos(3), tray_pos(3), suction_on(1)]
```

### Action Space — 4 dimensions

```
[dx, dy, dz, suction_action]
```
- `dx, dy, dz ∈ [-0.05, 0.05]` metres — end-effector delta, resolved via IK
- `suction_action > 0` — attempt to activate suction (only works if within 8 cm of cube)
- `suction_action ≤ 0` — deactivate suction

### Reward Function

```
r = -dist(EE, cube) - dist(cube, tray)
    + 10   (one-time: first successful suction grasp)
    + 50   (one-time: cube placed within 5 cm of tray)
```

The dense components (`-dist`) guide approach behaviour. The one-time bonuses are anti-gaming: they fire only once per episode so the agent cannot repeatedly collect them.

### Suction Mechanism

Suction is modelled using a PyBullet `JOINT_FIXED` constraint between the end-effector link and the cube body. It activates only when `dist(EE, cube) < 0.08 m` and a positive suction command is issued.

---

## Training Setup

| Parameter | Value |
|---|---|
| **Algorithm** | SAC (Soft Actor-Critic) |
| **Library** | Stable-Baselines3 |
| **Steps per seed** | 1,200,000 |
| **Learning rate** | 3e-4 |
| **Replay buffer** | 1,000,000 transitions |
| **Batch size** | 256 |
| **Entropy coefficient** | Auto-tuned (`ent_coef="auto"`) |
| **Episode length** | Max 500 steps |

---

## Results: 3-Seed Validation

Three independent seeds trained for 1.2M steps each, showing **consistent and reproducible learning**.

| Metric | Seed 0 | Seed 1 | Seed 2 | **Average** |
|---|---|---|---|---|
| **Total episodes** | 24,582 | 25,807 | 26,797 | 25,729 |
| **Final eval mean reward** | +37.7 | +34.2 | +34.1 | **+35.3** |
| **Peak eval mean reward** | +42.3 | +43.3 | +43.2 | **+42.9** |
| **Full success rate (reward > 40)** | 16.4% | 16.9% | 16.6% | **16.6%** |
| **Max episode reward** | +47.06 | +47.11 | +46.94 | **+47.04** |
| **Last 100 eps with positive reward** | 99/100 | 99/100 | 97/100 | **98%** |
| **First successful episode** | Ep 99 | Ep 58 | Ep 58 | ~Ep 72 |

### Learning Trajectory

All three seeds follow the same four-phase pattern:

1. **Exploration (0–50k steps)** — Random policy, reward ~ -400 to -100
2. **Discovery (50k–200k steps)** — First successes appear (episodes 58–99), entropy collapses then recovers
3. **Improvement (200k–600k steps)** — Success rate climbs, episode length drops from 500 to ~30–50 steps
4. **Stabilisation (600k–1.2M steps)** — Plateau at ~+35–43 eval reward, ~16–17% success rate

### What ~17% Success Rate Means

This is **no human demonstrations, no curriculum, no reward shaping tricks** — just a raw SAC agent with a well-designed action space. For reference:
- Direct joint control: **0%**
- SAC + Granger causal reward (Exp5): **72% peak** (required careful tuning)
- SAC + OT demo buffer (Exp19): **90% at 1M steps** (required pre-loaded buffer)

The suction gripper baseline establishes what plain SAC can achieve given a sensible action space.

---

## File Structure

```
.
├── run.sh                          ← automated pipeline (start here)
├── requirements.txt
├── pick_place_env_suction.py       ← PyBullet environment
├── train_suction.py                ← SAC training script
│
├── experiments/
│   ├── exp35_causal_ablation/      ← ablation: 4 methods × 2 tasks × 8 seeds
│   │   ├── train_ablation.py
│   │   ├── exp35_mechanistic_analysis.py
│   │   ├── recover_auto_alpha.py
│   │   ├── EXPERIMENT_REPORT.md
│   │   └── analysis/               ← 7 plots + seed_summary.csv
│   │
│   └── exp36_alpha_trajectory/     ← alpha bifurcation: 5 tasks × 8 seeds
│       ├── train_alpha_trajectory.py
│       └── launch_alpha_trajectory.sh
│
└── results/
    └── suction_gripper/
        ├── seed0/
        │   ├── eval/evaluations.npz
        │   ├── models/best_model/best_model.zip
        │   └── models/final_model.zip
        ├── seed1/
        └── seed2/
```

---

## Additional Experiments: Alpha Bifurcation Analysis

Beyond the main suction gripper results, we ran a mechanistic study on **why SAC succeeds on some seeds and fails on others** — a phenomenon observed across many MetaWorld tasks.

### The Core Finding: Alpha Bifurcation

SAC's auto-tuned entropy coefficient **α** (which controls exploration vs exploitation) behaves very differently depending on whether a seed will eventually succeed:

- **Seeds that solve the task**: α naturally decays toward ~0 as the agent discovers and exploits the solution. SAC's auto-entropy mechanism effectively acts as a self-annealing schedule.
- **Seeds that fail**: α stays elevated throughout training — the agent keeps exploring because it never finds a stable high-reward region to exploit.

This means **α trajectory is a leading indicator of learning success** — you can often predict whether a seed will succeed by watching how α evolves in the first 200–400k steps, well before the evaluation reward diverges.

### Exp35 — Causal Ablation Study

**64 runs**: 4 methods × 2 tasks × 8 seeds on MetaWorld (`peg-insert-side-v3`, `pick-place-v3`)

| Method | Entropy | Demo Reward |
|---|---|---|
| **A** — SAC baseline | Auto-tuned | None |
| **B** — SAC + anneal | Fixed schedule (0.1 → 0.005) | None |
| **C** — demo_smooth | Auto-tuned | Self-bootstrapped k-NN |
| **D** — demo_smooth + anneal | Fixed schedule | Self-bootstrapped k-NN |

Key finding: **Method A (plain SAC with auto-entropy) solves 7/8 seeds on peg-insert** — the additional complexity of manual annealing (B) or demo shaping (C, D) does not consistently improve over the baseline. The auto-entropy mechanism in SAC is already doing something close to optimal annealing on seeds that succeed.

Analysis plots are in `experiments/exp35_causal_ablation/analysis/`.

### Exp36 — Alpha Trajectory Logging

**40 runs**: 5 tasks × 8 seeds (Method A only), with α logged every 1,000 steps.

This experiment fixed the logging bug in Exp35 (where auto-α wasn't captured) and produced the full **α-bifurcation figure**: plotting α over training time, coloured by whether the seed ultimately succeeded, shows a clean split at roughly 200–400k steps.

Tasks: `peg-insert-side-v3`, `pick-place-v3`, `door-open-v3`, `drawer-close-v3`, `window-open-v3`

---

## How to Reproduce

### 1. Full pipeline (automated)
```bash
bash run.sh
```

### 2. Single training run
```bash
pip install -r requirements.txt
python train_suction.py --seed 0 --timesteps 1200000
```

### 3. Evaluate a saved model
```bash
python3 - <<'EOF'
from stable_baselines3 import SAC
from pick_place_env_suction import PickPlaceSuctionEnv

env = PickPlaceSuctionEnv()
model = SAC.load("results/suction_gripper/seed0/models/best_model/best_model.zip")
obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
env.close()
EOF
```

### 4. TensorBoard
```bash
tensorboard --logdir logs/sac_suction_seed0/tb
```

---

## Dependencies

```
numpy, gymnasium, pybullet, stable-baselines3
tensorboard, tqdm, rich
matplotlib, pandas, scikit-learn, torch
```

Full pinned versions in `requirements.txt`.
