# Reinforcement Learning for Robotic Pick-and-Place
### Suction Gripper Control with SAC + Alpha Bifurcation Analysis

A full reinforcement learning pipeline for robotic manipulation. A **Kuka IIWA arm with a suction gripper** learns to pick up a cube and place it in a tray using **Soft Actor-Critic (SAC)** in **PyBullet**. The project also includes a mechanistic analysis of SAC's entropy coefficient (`alpha`) as a diagnostic signal for predicting learning success before training converges.

---

## Quick Start

This repo now has **two automated shell entrypoints**, both intended to be run inside a **Docker container based on a bare Ubuntu image**.

### 1. Full training pipeline

```bash
bash run.sh
```

What it does:
- installs Ubuntu system dependencies
- creates or reuses `venv/`
- installs Python dependencies
- runs a smoke test on the environment
- trains the SAC agent for **3 seeds x 1,200,000 steps**

Outputs from `run.sh` are written to:

```text
logs/sac_suction_seed0/
logs/sac_suction_seed1/
logs/sac_suction_seed2/
```

Each seed directory contains evaluation checkpoints, models, TensorBoard logs, and diagnostics.

### 2. Showcase the already trained best model

```bash
bash test.sh
```

What it does:
- installs Ubuntu system dependencies needed for evaluation
- creates or reuses `venv/`
- installs Python dependencies
- automatically selects the **best saved SAC checkpoint** already included in this repo
- runs the trained policy
- if a GUI display is available, shows the rollout live in PyBullet
- otherwise records a headless MP4

`test.sh` uses the committed pretrained checkpoints under:

```text
results/suction_gripper/
```

and saves its generated outputs to:

```text
results/test_runs/
```

That output directory contains:
- `successful_rollout.mp4` or `best_rollout.mp4`
- `summary.json`

---

## Project Overview

### The Task

A **Kuka IIWA robot arm** in PyBullet must learn to:
1. **Reach** a randomly placed cube on a table
2. **Activate suction** when the end-effector is within 8 cm of the cube
3. **Lift and carry** the cube to a tray
4. **Place** the cube within 5 cm of the tray center

### Key Design Decision: IK-Resolved End-Effector Control

The main design choice that made this task learnable was switching from direct joint control to **end-effector delta control resolved via Inverse Kinematics (IK)**.

| | Direct Joint Control | **EE-Delta + IK (This Work)** |
|---|---|---|
| **Action space** | 8D: `[j1..j7, suction]` | **4D: `[dx, dy, dz, suction]`** |
| **What the agent learns** | IK mapping + manipulation simultaneously | **Just where to move** |

By letting PyBullet handle IK, the agent only needs to reason about *where to move the hand* rather than learning inverse kinematics implicitly.

---

## Environment Specification

### State Space - 24 dimensions

```text
[joint_angles(7), joint_velocities(7), ee_pos(3), object_pos(3), tray_pos(3), suction_on(1)]
```

### Action Space - 4 dimensions

```text
[dx, dy, dz, suction_action]
```

- `dx, dy, dz` in `[-0.05, 0.05]` meters
- `suction_action > 0` attempts to activate suction
- `suction_action <= 0` deactivates suction

### Reward Function

```text
r = -dist(EE, cube) - dist(cube, tray)
    + 10   (one-time: first successful suction grasp)
    + 50   (one-time: cube placed within 5 cm of tray)
```

### Suction Mechanism

Suction is modeled with a PyBullet `JOINT_FIXED` constraint between the end-effector and the cube. It only activates when `dist(EE, cube) < 0.08 m` and a positive suction command is issued.

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

Three independent seeds trained for 1.2M steps each, showing consistent and reproducible learning.

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

1. **Exploration (0-50k steps)** - random policy, reward about `-400` to `-100`
2. **Discovery (50k-200k steps)** - first successes appear, entropy collapses then recovers
3. **Improvement (200k-600k steps)** - success rate climbs, episode length drops sharply
4. **Stabilization (600k-1.2M steps)** - plateau at about `+35` to `+43` eval reward

---

## File Structure

```text
.
|-- run.sh                         # automated full training pipeline
|-- test.sh                        # automated best-model evaluation/showcase
|-- test_trained_model.py          # selects best saved checkpoint and records/plays rollout
|-- requirements.txt
|-- pick_place_env_suction.py      # PyBullet suction-gripper environment
|-- train_suction.py               # SAC training script
|
|-- experiments/
|   |-- exp35_causal_ablation/
|   |   |-- train_ablation.py
|   |   |-- exp35_mechanistic_analysis.py
|   |   |-- recover_auto_alpha.py
|   |   `-- analysis/
|   |
|   `-- exp36_alpha_trajectory/
|       |-- train_alpha_trajectory.py
|       `-- launch_alpha_trajectory.sh
|
`-- results/
    |-- suction_gripper/
    |   |-- seed0/
    |   |   |-- eval/evaluations.npz
    |   |   |-- models/best_model/best_model.zip
    |   |   `-- models/final_model.zip
    |   |-- seed1/
    |   `-- seed2/
    |
    |-- best_model_anim/
    `-- test_runs/                 # generated by test.sh when evaluation is run
```

---

## Additional Experiments: Alpha Bifurcation Analysis

Beyond the main suction gripper results, the repo also contains a mechanistic study on **why SAC succeeds on some seeds and fails on others**.

### The Core Finding: Alpha Bifurcation

SAC's auto-tuned entropy coefficient (`alpha`) behaves very differently depending on whether a seed will eventually succeed:

- **Successful seeds**: `alpha` naturally decays toward zero as the agent discovers and exploits the solution.
- **Failed seeds**: `alpha` stays elevated, indicating continued exploration without finding a stable high-reward strategy.

This makes `alpha` a useful early indicator of whether a training run is likely to solve the task.

### Exp35 - Causal Ablation Study

**64 runs** across 4 methods, 2 tasks, and 8 seeds on MetaWorld.

### Exp36 - Alpha Trajectory Logging

**40 runs** across 5 tasks and 8 seeds, with `alpha` logged every 1,000 steps.

---

## How to Reproduce

### 1. Full training from scratch inside a bare Ubuntu Docker container

```bash
bash run.sh
```

This installs dependencies, verifies the environment, and trains the SAC agent for 3 seeds.

### 2. Run the already trained best model inside a bare Ubuntu Docker container

```bash
bash test.sh
```

This loads the best saved SAC checkpoint already present in the repo, runs it, and saves outputs under:

```text
results/test_runs/
```

If no GUI display is available, it records an MP4 there automatically.

### 3. Single training run

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 train_suction.py --seed 0 --timesteps 1200000
```

### 4. Programmatic evaluation of a saved model

```bash
python3 test_trained_model.py --headless
```

### 5. TensorBoard for a training run

```bash
tensorboard --logdir logs/sac_suction_seed0/tb
```

---

## Dependencies

```text
numpy, gymnasium, pybullet, stable-baselines3
tensorboard, tqdm, rich
matplotlib, pandas, scikit-learn, torch
```

Full versions are listed in `requirements.txt`.
