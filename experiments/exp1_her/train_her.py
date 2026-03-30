"""
Experiment 1: HER + SAC
------------------------
Hindsight Experience Replay with Soft Actor-Critic.

HER converts every failed trajectory into a successful one by
replacing the desired goal with the achieved goal. This makes
sparse reward signals learnable.

Usage:
    python train_her.py
    python train_her.py --timesteps 1000000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from pick_place_env_her import PickPlaceHEREnv


def make_env(render=False):
    return PickPlaceHEREnv(render_mode="human" if render else None)


def train(timesteps, render):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_vec_env(lambda: make_env(render), n_envs=1)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.95,           # shorter horizon for goal-conditioned
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        learning_starts=1000,       # HER needs >= 1 full episode before sampling
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
            name_prefix="her_sac",
        ),
    ]

    print(f"\n[EXP1 — HER+SAC] Training for {timesteps:,} steps...\n")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    model.save("models/final")
    print(f"\nDone. Model → models/final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args.timesteps, args.render)
