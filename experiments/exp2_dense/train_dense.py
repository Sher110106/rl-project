"""
Experiment 2: Dense Gripper Shaping + SAC
------------------------------------------
Same tuned SAC as v4, but with the dense gripper env that adds
explicit gripper-closure reward near the cube.

Usage:
    python train_dense.py
    python train_dense.py --timesteps 2000000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from pick_place_env_dense import PickPlaceDenseEnv


def make_env(render=False):
    return PickPlaceDenseEnv(render_mode="human" if render else None)


def train(timesteps, render):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_vec_env(lambda: make_env(render), n_envs=1)

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
        verbose=1,
        tensorboard_log="logs/tb/",
    )

    eval_env = make_vec_env(lambda: make_env(False), n_envs=1)
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="models/best/",
            log_path="logs/eval/",
            eval_freq=10_000,
            n_eval_episodes=5,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/checkpoints/",
            name_prefix="dense_sac",
        ),
    ]

    print(f"\n[EXP2 — Dense Gripper + SAC] Training for {timesteps:,} steps...\n")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    model.save("models/final")
    print(f"\nDone. Model → models/final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args.timesteps, args.render)
