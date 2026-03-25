# Usage Guide

This document explains exactly how to set up and run this repository.

## 1) What each main file is for

- `train.py`: the executable entry point. You run this file to train or evaluate RL agents.
- `pick_place_env.py`: a custom Gymnasium environment definition used by `train.py`.
- `requirements.txt`: Python dependencies for installation with `pip`.
- `results/RESULTS.md`: experiment notes and historical outcomes (not code execution logic).

## 2) One-time setup

Run from the repo root:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Minimal quick start

Train default agent (SAC, 100k steps):

```powershell
python train.py
```

Then visualize the trained best model:

```powershell
python train.py --algo sac --enjoy
```

## 4) What command does what

### Train commands

```powershell
python train.py
```
- Trains with default settings:
- Algorithm: `sac`
- Timesteps: `100000`
- GUI during training: off

```powershell
python train.py --algo sac
python train.py --algo td3
python train.py --algo ddpg
```
- Trains with the selected algorithm.

```powershell
python train.py --timesteps 200000
```
- Trains for 200k steps (default algorithm unless `--algo` is also set).

```powershell
python train.py --algo td3 --timesteps 300000
```
- Combined control of algorithm and training length.

```powershell
python train.py --render
```
- Trains while opening PyBullet GUI.
- Usually much slower than headless training.

### Evaluation / "watch the robot" command

```powershell
python train.py --algo sac --enjoy
```
- Loads `models/sac_pick_place_best/best_model.zip`.
- Opens GUI and runs deterministic policy continuously.
- Prints episode reward after each episode.

You can swap algorithm:

```powershell
python train.py --algo td3 --enjoy
python train.py --algo ddpg --enjoy
```

If the expected model does not exist, script prints:
- `No saved model at models/<algo>_pick_place_best/best_model.zip - train first.`

## 5) CLI flags in `train.py`

- `--algo {sac,td3,ddpg}`
- Selects RL algorithm class from Stable-Baselines3.

- `--timesteps <int>`
- Number of environment steps for `model.learn(...)`.

- `--render`
- Enables human GUI mode in the training environment.

- `--enjoy`
- Skips training and runs the saved best policy in GUI mode.

Decision logic:
- If `--enjoy` is passed: run `enjoy(...)`.
- Else: run `train(...)`.

## 6) What happens internally when `train.py` runs

High-level flow:

1. Parse command-line args.
2. Import `PickPlaceEnv` from `pick_place_env.py`.
3. Build vectorized environment with `make_vec_env(...)`.
4. Create model (`SAC`, `TD3`, or `DDPG`) with `MlpPolicy`.
5. Attach callbacks:
- `EvalCallback` for periodic evaluation and best-model saving.
- `CheckpointCallback` for periodic checkpoints.
6. Train via `model.learn(...)`.
7. Save final model.

Generated artifacts:

- `models/<algo>_pick_place/final.zip`
- `models/<algo>_pick_place_best/best_model.zip`
- `models/<algo>_pick_place_checkpoints/*.zip`
- `logs/<algo>/...`
- `logs/<algo>_eval/...`

## 7) What `pick_place_env.py` does (and does not do)

`pick_place_env.py` defines the environment class `PickPlaceEnv`:

- Action space: 7D continuous joint targets in `[-1, 1]`.
- Observation space: 23D vector:
- 7 joint angles
- 7 joint velocities
- 3 end-effector coordinates
- 3 object coordinates
- 3 tray coordinates
- Reward:
- `-dist(ee, object) - dist(object, tray)`
- `+10` one-time grasp bonus when close-and-lift condition is met
- `+50` success bonus when object is near tray center
- Episode end:
- `terminated=True` on success
- `truncated=True` when max steps (`500`) is reached

It is normally imported by `train.py`. Running `python pick_place_env.py` directly does not start training because there is no main execution block in that file.

## 8) How to actually see learned behavior

1. Train first:

```powershell
python train.py --algo sac --timesteps 200000
```

2. Watch policy:

```powershell
python train.py --algo sac --enjoy
```

What you should expect:
- A PyBullet GUI window opens.
- Robot repeatedly attempts the task episode by episode.
- Episode rewards are printed in terminal.

Important practical note:
- If training is short, you may only see approach behavior, not full successful pick-and-place.
- For complex 7-DoF manipulation, longer runs and reward shaping tweaks are often needed.

## 9) Optional monitoring with TensorBoard

After or during training:

```powershell
tensorboard --logdir logs
```

Then open the shown local URL in your browser.

## 10) Recommended workflow

1. `pip install -r requirements.txt`
2. `python train.py --algo sac --timesteps 100000`
3. `python train.py --algo sac --enjoy`
4. Increase timesteps and compare algorithms (`td3`, `ddpg`) if needed.
