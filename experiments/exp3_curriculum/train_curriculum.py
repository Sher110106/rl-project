"""
Experiment 3: Curriculum Learning + SAC
----------------------------------------
Progressive difficulty with auto-advancing stages.

Stage 1: Cube near tray, EE above cube → learn grasp + place
Stage 2: Cube random, EE above cube   → learn reach + grasp
Stage 3: Full random                   → full task

Advances when success rate > 15% over last 100 episodes.

Usage:
    python train_curriculum.py
    python train_curriculum.py --timesteps 2000000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from pick_place_env_curriculum import PickPlaceCurriculumEnv


def make_env(render=False):
    return PickPlaceCurriculumEnv(render_mode="human" if render else None)


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
            n_eval_episodes=10,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/checkpoints/",
            name_prefix="curriculum_sac",
        ),
    ]

    print(f"\n[EXP3 — Curriculum + SAC] Training for {timesteps:,} steps...\n")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    model.save("models/final")
    print(f"\nDone. Model → models/final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args.timesteps, args.render)
