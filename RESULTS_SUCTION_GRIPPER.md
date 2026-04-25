# Suction Gripper Pick-and-Place RL

This branch contains the full implementation and results for training a **SAC agent** on a **PyBullet pick-and-place task with a suction gripper**.

## What This Is

A reinforcement learning environment where a **Kuka IIWA robot arm** learns to:
1. **Reach** a randomly placed cube
2. **Activate suction** when close enough (distance < 8 cm)
3. **Lift and carry** the cube to a tray
4. **Place** the cube at the tray position

## Key Design Decision: IK-Resolved Control

Unlike the initial attempt with direct joint control (which failed completely), this implementation uses **end-effector delta control resolved via Inverse Kinematics (IK)**.

| | Direct Joint Control (Failed) | **EE-Delta + IK (This Branch)** |
|---|---|---|
| **Action Space** | 8D: `[j1..j7, suction]` | **4D: `[dx, dy, dz, suction]`** |
| **Control** | Agent sets each joint angle | **Agent commands gripper movement** |
| **Complexity** | Must learn implicit IK | **PyBullet handles IK** |
| **Result** | 0% success | **16.6% success rate** |

The action space directly maps to the task: "move hand here, then suction".

---

## Environment Specification

### State Space (24-dim)
```
[joint_angles(7) + joint_velocities(7) + ee_pos(3) + object_pos(3) + tray_pos(3) + suction_on(1)]
```

### Action Space (4-dim)
```
[dx, dy, dz, suction_action]
```
- `dx, dy, dz`: End-effector delta movement, clipped to `[-0.05, 0.05]` meters
- `suction_action > 0`: Attempt to activate suction (if within 8 cm of cube)
- `suction_action <= 0`: Deactivate suction

### Reward Function
```
reward = -dist(EE, cube) - dist(cube, tray)
         + 10 (one-time: first successful suction activation)
         + 50 (one-time: cube placed within 5 cm of tray)
```

### Suction Mechanism
- PyBullet `JOINT_FIXED` constraint between the EE link and the cube
- Activates only when `dist(EE, cube) < 0.08` meters
- Deactivates on command or at episode end

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| **Algorithm** | SAC (Soft Actor-Critic) |
| **Library** | Stable-Baselines3 |
| **Total Steps** | 1,200,000 per seed |
| **Episodes** | ~24,000–27,000 per seed |
| **Episode Length** | Max 500 steps |
| **Learning Rate** | 3e-4 |
| **Buffer Size** | 1,000,000 |
| **Batch Size** | 256 |
| **Entropy** | Auto-tuned (`ent_coef="auto"`) |
| **Device** | GPU (GPU) |
| **Speed** | ~37–38 steps/second |
| **Wall Time** | ~8.5 hours per seed |

---

## Results: 3-Seed Validation

All 3 seeds completed 1.2M steps and show **consistent, stable learning**.

| Metric | Seed 0 | Seed 1 | Seed 2 | **Average** |
|--------|--------|--------|--------|-------------|
| **Total Episodes** | 24,582 | 25,807 | 26,797 | 25,729 |
| **Final Eval Mean** | +37.7 | +34.2 | +34.1 | **+35.3** |
| **Peak Eval Mean** | +42.3 | +43.3 | +43.2 | **+42.9** |
| **Success Rate (>40 reward)** | 16.4% | 16.9% | 16.6% | **16.6%** |
| **Max Episode Reward** | +47.06 | +47.11 | +46.94 | **+47.04** |
| **Last 100 Eps (positive)** | 99/100 | 99/100 | 97/100 | **98%** |
| **First Success** | Episode 99 | Episode 58 | Episode 58 | ~Episode 72 |

### Interpretation
- **~17% of episodes achieve full success** (reward > 40 = +50 success bonus minus early penalties)
- **~98% of recent episodes have positive reward** — the agent consistently gets close to or achieves the goal
- **Max reward ~+47** — very close to the theoretical maximum (+50 success + small dense rewards)
- **Stable convergence** — no collapse at the end; eval rewards maintained at ~+35–43

### Learning Trajectory
All seeds follow the same pattern:
1. **Early (0–50k steps)**: Random exploration, reward ~ -400 to -100
2. **Discovery (50k–200k)**: First successes appear (episode 58–99), entropy collapses then recovers
3. **Improvement (200k–600k)**: Success rate climbs, episode length drops from 500 to ~30–50 steps
4. **Stabilization (600k–1.2M)**: Plateau at ~+35–43 eval reward, ~16–17% success rate

---

## File Structure

### Code
```
pick_place_env_suction.py   # PyBullet environment (suction + IK)
train_suction.py             # SAC training script
```

### Results
```
results/suction_gripper/
├── seed0/
│   ├── eval/evaluations.npz              # 120 eval checkpoints (SB3 format)
│   ├── models/final_model.zip            # Final 1.2M-step model
│   ├── models/best_model/best_model.zip  # Best eval checkpoint
│   └── tb/events.out.tfevents.*          # TensorBoard logs
├── seed1/
│   └── (same structure)
└── seed2/
    └── (same structure)
```

### Full Diagnostics (On Remote Server)

```
~/projects/rlp/logs/sac_suction_seed{0,1,2}/diagnostics/episode_diagnostics.csv
```

Use the `fetch_diagnostics.sh` script to download them locally if needed.

---

## How to Reproduce

### 1. Environment Setup
```bash
pip install gymnasium numpy pybullet stable-baselines3
```

### 2. Quick Test
```bash
python -c "
from pick_place_env_suction import PickPlaceSuctionEnv
env = PickPlaceSuctionEnv()
obs, _ = env.reset()
print('State shape:', obs.shape)  # (24,)
print('Action space:', env.action_space.shape)  # (4,)
"
```

### 3. Train from Scratch
```bash
python train_suction.py --seed 0 --timesteps 1200000
```

### 4. Evaluate a Saved Model
```bash
python -c "
from stable_baselines3 import SAC
from pick_place_env_suction import PickPlaceSuctionEnv

env = PickPlaceSuctionEnv()
model = SAC.load('results/suction_gripper/seed0/models/best_model/best_model.zip')

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
"
```

### 5. View TensorBoard
```bash
tensorboard --logdir results/suction_gripper/seed0/tb
```

---

## Key Takeaways

1. **IK is essential for manipulation RL**: Direct joint control (8D) completely failed. EE-delta control (4D) with IK resolution made the task learnable.
2. **Suction gripper works**: The PyBullet `JOINT_FIXED` constraint successfully models vacuum suction, enabling reliable pick-and-carry.
3. **~17% success is meaningful**: In a sparse-to-dense reward setup with no human demonstrations, consistent ~17% success across 3 seeds shows the method works.
4. **Entropy auto-tuning is sufficient**: No manual annealing schedule needed. SAC's adaptive `ent_coef` naturally explores early and exploits later.

---

## Comparison with Prior Work

| Experiment | Control | Reward | Success Rate | Notes |
|------------|---------|--------|--------------|-------|
| **v1 SAC (baseline)** | 7D joints | `-dist(EE,cube) - dist(cube,tray)` | 0% | Never solved |
| **v4 + Dense Gripper** | 4D EE-delta | +gripper shaping | ~2% | Best prior PyBullet result |
| **v4 + Granger (Exp5)** | 4D EE-delta | +causal reward | 72% peak | Required careful tuning |
| **v4 + OT Demo (Exp19)** | 4D EE-delta | +demo reward | 90% at 1M | Required pre-loaded buffer |
| **This Work** | **4D EE-delta + suction** | Simple dense | **16.6%** | **No human data, no complex shaping** |

This establishes a **minimal viable baseline** for suction-based manipulation: IK-resolved control + simple dense reward + SAC = consistent but modest success.

---

## Next Steps / Extensions

- **Dense suction shaping**: Reward proximity × suction activation (like v4 gripper shaping)
- **Self-bootstrapped demo buffer**: Add OT-style nearest-neighbor reward from successful episodes
- **Longer training**: 2–3M steps may improve success rate beyond 17%
- **Curriculum**: Start with cube near tray, progressively randomize
- **Multi-object**: Multiple cubes, sorting task
- **Different robots**: Franka Panda, UR5 with suction

---



  

