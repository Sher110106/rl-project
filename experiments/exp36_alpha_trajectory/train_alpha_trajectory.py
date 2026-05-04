#!/usr/bin/env python3
"""
exp36_alpha_trajectory.py
=========================
Re-run SAC baseline (Method A) with full α trajectory logging every 1k steps.

Key addition over exp35: AlphaTrajectoryCallback logs the actual learned
entropy coefficient at every 1000 steps, enabling the α-bifurcation figure.

Usage:
    python train_alpha_trajectory.py --task peg-insert-side-v3 --seed 0
    python train_alpha_trajectory.py --task door-open-v3 --seed 3 --steps 1000000

Outputs (per run):
    alpha_trajectory.csv     ← α at every 1k steps  [THE NEW KEY LOG]
    eval/evaluations.npz     ← eval reward every 10k steps
    first_success_step.txt   ← first step with episode reward >= 500
    buffer_success_log.csv   ← replay buffer success fraction every 50k steps
    qvalue_probe_log.csv     ← Q(s,a) at probe states per eval checkpoint
    success_log.csv          ← per-episode success + reward
    final_model.zip          ← saved model at end of training
"""

import argparse
import csv
import os

import gymnasium as gym
import metaworld
import numpy as np
import torch as th
from sklearn.neighbors import BallTree
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SUCCESS_THRESHOLD = 500.0
NEAR_OBJECT_DIST  = 0.05
EVAL_FREQ         = 10_000
N_EVAL_EPISODES   = 20
BUFFER_LOG_EVERY  = 50_000
ALPHA_LOG_EVERY   = 1_000


# ─────────────────────────────────────────────────────────────
# CALLBACK 1: Alpha Trajectory Logger  ← THE KEY NEW PIECE
# ─────────────────────────────────────────────────────────────
class AlphaTrajectoryCallback(BaseCallback):
    """
    Logs the learned entropy coefficient α = exp(log_ent_coef) every
    ALPHA_LOG_EVERY steps. Produces alpha_trajectory.csv.

    Verification: final α here should match
        SAC.load('final_model.zip').log_ent_coef.exp().item()
    """
    def __init__(self, logdir: str):
        super().__init__()
        self.logdir = logdir
        self.records = []
        self._csv_path = os.path.join(logdir, "alpha_trajectory.csv")

    def _on_step(self) -> bool:
        if self.num_timesteps % ALPHA_LOG_EVERY == 0:
            alpha = self.model.log_ent_coef.exp().item()
            self.records.append({"step": self.num_timesteps, "alpha": alpha})
        return True

    def _on_training_end(self):
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "alpha"])
            writer.writeheader()
            writer.writerows(self.records)


# ─────────────────────────────────────────────────────────────
# CALLBACK 2: Mechanistic Logging (carried over from exp35)
# ─────────────────────────────────────────────────────────────
class MechanisticCallback(BaseCallback):
    """
    Logs:
      - Per-episode success + reward (success_log.csv)
      - First success step (first_success_step.txt)
      - Replay buffer success fraction every 50k steps (buffer_success_log.csv)
      - Q-values at probe states at each eval checkpoint (qvalue_probe_log.csv)
    """
    def __init__(self, logdir: str):
        super().__init__()
        self.logdir = logdir
        self._first_success_logged = False
        self._last_buf_log = 0
        self._probe_states = None
        self._probe_actions = None
        self._probe_saved_step = None

        # Files opened on training start
        self._success_f = None
        self._success_writer = None
        self._buf_f = None
        self._buf_writer = None
        self._qv_f = None
        self._qv_writer = None

    def _on_training_start(self):
        self._success_f = open(os.path.join(self.logdir, "success_log.csv"), "w", newline="")
        self._success_writer = csv.DictWriter(
            self._success_f, fieldnames=["episode", "step", "reward", "success"])
        self._success_writer.writeheader()

        self._buf_f = open(os.path.join(self.logdir, "buffer_success_log.csv"), "w", newline="")
        self._buf_writer = csv.DictWriter(
            self._buf_f, fieldnames=["step", "buffer_success_fraction"])
        self._buf_writer.writeheader()

        self._qv_f = open(os.path.join(self.logdir, "qvalue_probe_log.csv"), "w", newline="")
        self._qv_writer = csv.DictWriter(
            self._qv_f, fieldnames=["step", "mean_q", "std_q", "n_probes"])
        self._qv_writer.writeheader()

    def _on_step(self) -> bool:
        step = self.num_timesteps
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        # Per-episode logging
        if "episode" in info:
            ep_info = info["episode"]
            reward = ep_info.get("r", 0.0)
            ep_num = getattr(self, "_ep_count", 0) + 1
            self._ep_count = ep_num
            success = int(reward >= SUCCESS_THRESHOLD)
            self._success_writer.writerow({
                "episode": ep_num, "step": step,
                "reward": round(reward, 2), "success": success
            })

            # First success
            if success and not self._first_success_logged:
                self._first_success_logged = True
                with open(os.path.join(self.logdir, "first_success_step.txt"), "w") as f:
                    f.write(str(step))

        # Buffer success fraction
        if step - self._last_buf_log >= BUFFER_LOG_EVERY and step > 0:
            self._last_buf_log = step
            frac = self._compute_buffer_success_fraction()
            self._buf_writer.writerow({
                "step": step, "buffer_success_fraction": round(frac, 6)})
            self._buf_f.flush()

        # Save probe states once (at first success or 300k)
        if self._probe_states is None:
            should_save = (
                (self._first_success_logged and self._probe_saved_step is None) or
                (step >= 300_000 and self._probe_saved_step is None)
            )
            if should_save:
                self._save_probe_states()

        return True

    def _compute_buffer_success_fraction(self) -> float:
        buf = self.model.replay_buffer
        if buf.size() == 0:
            return 0.0
        n = buf.size()
        # SB3 replay buffer stores rewards in buf.rewards[:n]
        rewards = buf.rewards[:n].flatten()
        return float((rewards >= SUCCESS_THRESHOLD).mean())

    def _save_probe_states(self):
        buf = self.model.replay_buffer
        n = min(buf.size(), 20)
        if n == 0:
            return
        # Sample up to 20 recent successful transitions
        rewards = buf.rewards[:buf.size()].flatten()
        success_idxs = np.where(rewards >= SUCCESS_THRESHOLD)[0]
        if len(success_idxs) == 0:
            success_idxs = np.arange(buf.size())
        chosen = success_idxs[np.random.choice(len(success_idxs),
                                                min(20, len(success_idxs)),
                                                replace=False)]
        obs = buf.observations[chosen].reshape(len(chosen), -1)
        acts = buf.actions[chosen].reshape(len(chosen), -1)
        self._probe_states = obs
        self._probe_actions = acts
        self._probe_saved_step = self.num_timesteps
        np.save(os.path.join(self.logdir, "probe_states.npy"), obs)
        np.save(os.path.join(self.logdir, "probe_actions.npy"), acts)

    def log_qvalue_at_probes(self, step: int):
        if self._probe_states is None or len(self._probe_states) == 0:
            self._qv_writer.writerow({"step": step, "mean_q": -1.0, "std_q": 0.0, "n_probes": 0})
            return
        device = self.model.device
        obs_t = th.tensor(self._probe_states, dtype=th.float32, device=device)
        act_t = th.tensor(self._probe_actions, dtype=th.float32, device=device)
        with th.no_grad():
            q1, q2 = self.model.critic(obs_t, act_t)
            q_vals = th.min(q1, q2).cpu().numpy().flatten()
        self._qv_writer.writerow({
            "step": step,
            "mean_q": round(float(q_vals.mean()), 4),
            "std_q": round(float(q_vals.std()), 4),
            "n_probes": len(q_vals)
        })
        self._qv_f.flush()

    def _on_training_end(self):
        self._success_f.close()
        self._buf_f.close()
        self._qv_f.close()


# ─────────────────────────────────────────────────────────────
# CALLBACK 3: Eval with Q-probe logging
# ─────────────────────────────────────────────────────────────
class AlphaEvalCallback(EvalCallback):
    """EvalCallback that also triggers Q-probe logging after each eval."""
    def __init__(self, mech_cb: MechanisticCallback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mech_cb = mech_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        # After each eval, log Q-values at probes
        if self.num_timesteps % EVAL_FREQ == 0:
            self.mech_cb.log_qvalue_at_probes(self.num_timesteps)
        return result


# ─────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────
def make_env(task_name: str, seed: int):
    ml1 = metaworld.ML1(task_name, seed=seed)
    env_cls = ml1.train_classes[task_name]
    task = ml1.train_tasks[seed % len(ml1.train_tasks)]

    def _init():
        env = env_cls()
        env.set_task(task)
        env = Monitor(env)
        return env

    return _init


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="peg-insert-side-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--logdir", type=str,
                        default="experiments/exp36_alpha_trajectory/logs")
    args = parser.parse_args()

    run_id = f"{args.task}__seed{args.seed}"
    logdir = os.path.join(args.logdir, run_id)
    os.makedirs(os.path.join(logdir, "eval"), exist_ok=True)

    print(f"[START] task={args.task} seed={args.seed} steps={args.steps}")

    # Environments
    train_env = DummyVecEnv([make_env(args.task, args.seed)])
    eval_env  = DummyVecEnv([make_env(args.task, args.seed)])

    # Callbacks
    alpha_cb = AlphaTrajectoryCallback(logdir=logdir)
    mech_cb  = MechanisticCallback(logdir=logdir)
    eval_cb  = AlphaEvalCallback(
        mech_cb=mech_cb,
        eval_env=eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        log_path=os.path.join(logdir, "eval"),
        best_model_save_path=None,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # SAC — Method A: auto-entropy, no demo reward, CPU
    model = SAC(
        "MlpPolicy",
        train_env,
        batch_size=256,
        buffer_size=1_000_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy="auto",
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,
        device="cpu",
    )

    model.learn(
        total_timesteps=args.steps,
        callback=[alpha_cb, mech_cb, eval_cb],
        reset_num_timesteps=True,
    )

    # Save final model (for α verification)
    model.save(os.path.join(logdir, "final_model"))

    # Verify: α from callback should match α from loaded model
    final_alpha_cb = alpha_cb.records[-1]["alpha"] if alpha_cb.records else None
    loaded = SAC.load(os.path.join(logdir, "final_model"), device="cpu")
    final_alpha_model = loaded.log_ent_coef.exp().item()
    match = (final_alpha_cb is not None and
             abs(final_alpha_cb - final_alpha_model) < 1e-4)
    print(f"[VERIFY] α from callback={final_alpha_cb:.6f}  "
          f"α from model={final_alpha_model:.6f}  match={match}")

    fss_path = os.path.join(logdir, "first_success_step.txt")
    fss = open(fss_path).read().strip() if os.path.exists(fss_path) else "never"
    print(f"[DONE] first_success={fss}")
    print(f"[COMPLETE] {run_id}")


if __name__ == "__main__":
    main()
