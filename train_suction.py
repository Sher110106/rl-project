"""
train_suction.py — SAC trainer for PickPlaceSuctionEnv with per-seed logging
"""
import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from pick_place_env_suction import PickPlaceSuctionEnv


def make_env(render=False, log_dir="logs/diagnostics"):
    return PickPlaceSuctionEnv(render_mode="human" if render else None, log_dir=log_dir)


def train(seed=0, timesteps=1_200_000, render=False):
    log_root = f"logs/sac_suction_seed{seed}"
    eval_dir = os.path.join(log_root, "eval")
    diag_dir = os.path.join(log_root, "diagnostics")
    model_dir = os.path.join(log_root, "models")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = make_vec_env(lambda: make_env(render, log_dir=diag_dir), n_envs=1)
    eval_env = make_vec_env(lambda: make_env(False, log_dir=diag_dir), n_envs=1)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        tensorboard_log=os.path.join(log_root, "tb"),
        seed=seed,
        verbose=1,
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=eval_dir,
            eval_freq=10_000,
            n_eval_episodes=5,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=os.path.join(model_dir, "checkpoints"),
            name_prefix=f"sac_suction_s{seed}",
        ),
    ]

    print(f"[train_suction] Starting SAC for {timesteps:,} steps, seed={seed}")
    print(f"[train_suction] Logs: {log_root}")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(model_dir, "final_model"))
    print("[train_suction] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_200_000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.timesteps, args.render)
