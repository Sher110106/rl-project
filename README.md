# Suction Gripper Pick-and-Place RL

Reinforcement learning for robotic pick-and-place using a **Kuka IIWA arm with a suction gripper** in PyBullet. Three independent SAC seeds trained for 1.2M steps each, achieving consistent ~16.6% success.

---

## Quick Links

- **Code**: [`pick_place_env_suction.py`](pick_place_env_suction.py), [`train_suction.py`](train_suction.py)
- **Results**: [`RESULTS_SUCTION_GRIPPER.md`](RESULTS_SUCTION_GRIPPER.md) — full analysis
- **Models**: [`results/suction_gripper/`](results/suction_gripper/) — eval data, checkpoints, tensorboard

---

## What This Is

A **PyBullet** environment where a robot arm learns to:

1. **Reach** a randomly placed cube
2. **Activate suction** when within 8 cm of the cube
3. **Lift and carry** the cube to a tray
4. **Place** the cube at the tray position (within 5 cm)

**Key design decision**: End-effector delta control resolved via **Inverse Kinematics (IK)** — the agent commands `[dx, dy, dz, suction]` and PyBullet handles joint angles.

---

## Environment

### State (24-dim)
```
[joint_angles(7) + joint_velocities(7) + ee_pos(3) + object_pos(3) + tray_pos(3) + suction_on(1)]
```

### Action (4-dim)
```
[dx, dy, dz, suction_action]
```
- `dx, dy, dz ∈ [-0.05, 0.05]` m — end-effector delta movement
- `suction_action > 0` — attempt to activate suction (if within 8 cm)
- `suction_action <= 0` — deactivate suction

### Reward
```
reward = -dist(EE, cube) - dist(cube, tray)
         + 10 (one-time: first successful suction)
         + 50 (one-time: cube placed at tray)
```

---

## Results: 3-Seed Validation



| Metric | Seed 0 | Seed 1 | Seed 2 | **Average** |
|--------|--------|--------|--------|-------------|
| **Total Episodes** | 24,582 | 25,807 | 26,797 | 25,729 |
| **Final Eval Mean** | +37.7 | +34.2 | +34.1 | **+35.3** |
| **Peak Eval Mean** | +42.3 | +43.3 | +43.2 | **+42.9** |
| **Success Rate (>40)** | 16.4% | 16.9% | 16.6% | **16.6%** |
| **Max Episode Reward** | +47.06 | +47.11 | +46.94 | **+47.04** |
| **Last 100 Eps Positive** | 99/100 | 99/100 | 97/100 | **98%** |
| **First Success** | Ep 99 | Ep 58 | Ep 58 | ~Ep 72 |

### Interpretation
- **~17% full success**: The agent consistently achieves the +50 placement bonus
- **~98% recent positive reward**: Almost all episodes are near-success or better
- **Stable across seeds**: Not a lucky seed — reproducible learning

---

## Why This Works (vs Direct Joint Control)

| | Direct Joints (Failed) | **EE-Delta + IK (This)** |
|---|---|---|
| **Action** | 8D: `[j1..j7, suction]` | **4D: `[dx, dy, dz, suction]`** |
| **Agent must learn** | IK + manipulation simultaneously | **Just where to move** |
| **Success rate** | **0%** | **16.6%** |
| **Eval reward** | ~0 | **+35** |

---

## How to Use

### 1. Setup
```bash
pip install gymnasium numpy pybullet stable-baselines3
```

### 2. Quick Test
```bash
python -c "
from pick_place_env_suction import PickPlaceSuctionEnv
env = PickPlaceSuctionEnv()
obs, _ = env.reset()
print('State:', obs.shape, 'Action:', env.action_space.shape)
"
```

### 3. Train
```bash
python train_suction.py --seed 0 --timesteps 1200000
```

### 4. Evaluate Saved Model
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

### 5. TensorBoard
```bash
tensorboard --logdir results/suction_gripper/seed0/tb
```

---

## File Structure

```
.
├── pick_place_env_suction.py      # Environment implementation
├── train_suction.py               # SAC training script
├── RESULTS_SUCTION_GRIPPER.md     # Full detailed analysis

└── results/suction_gripper/
    ├── seed0/
    │   ├── eval/evaluations.npz              # 120 eval checkpoints
    │   ├── models/final_model.zip            # Final 1.2M model
    │   ├── models/best_model/best_model.zip  # Best eval checkpoint
    │   └── tb/events.out.tfevents.*          # TensorBoard logs
    ├── seed1/
    └── seed2/
```



---

## Training Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | SAC (Soft Actor-Critic) |
| **Library** | Stable-Baselines3 |
| **Steps** | 1,200,000 |
| **LR** | 3e-4 |
| **Buffer** | 1,000,000 |
| **Batch** | 256 |
| **Entropy** | Auto-tuned |

| **Speed** | ~37–38 steps/sec |
| **Wall Time** | ~8.5 hours/seed |


---

## Comparison with Prior Work

| Experiment | Control | Extra Reward | Success | Notes |
|------------|---------|--------------|---------|-------|
| **v1 SAC (baseline)** | 7D joints | Dense dist | 0% | Never solved |
| **v4 + Dense Gripper** | 4D EE-delta | Gripper shaping | ~2% | Best prior PyBullet |
| **v4 + Granger (Exp5)** | 4D EE-delta | Causal reward | 72% peak | Required careful tuning |
| **v4 + OT Demo (Exp19)** | 4D EE-delta | Demo reward | 90% at 1M | Required pre-loaded buffer |
| **This Work** | **4D EE-delta + suction** | **Simple dense only** | **16.6%** | **No human data, no shaping** |

This establishes a **minimal viable baseline** for suction-based manipulation.

---

## Next Steps

- **Dense suction shaping**: Reward proximity × suction activation
- **Self-bootstrapped demo buffer**: Add OT-style nearest-neighbor reward
- **Longer training**: 2–3M steps for higher success rate
- **Curriculum**: Progressive difficulty
- **Multi-object**: Sorting tasks

---




