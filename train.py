"""
train.py
--------
Train a pick-and-place policy using SAC, TD3, or DDPG.

Usage:
    python train.py --algo sac              # default
    python train.py --algo td3
    python train.py --algo ddpg
    python train.py --algo sac --render     # open GUI while training (slow)
    python train.py --algo sac --timesteps 200000

After training, the best model is saved to:
    models/<algo>_pick_place_best/

To watch the trained policy:
    python train.py --algo sac --enjoy
"""

import argparse
import os

from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from pick_place_env import PickPlaceEnv

ALGOS = {"sac": SAC, "td3": TD3, "ddpg": DDPG}


def make_env(render=False):
    return PickPlaceEnv(render_mode="human" if render else None)


def train(algo_name, timesteps, render):
    AlgoCls = ALGOS[algo_name]
    model_dir = f"models/{algo_name}_pick_place"
    os.makedirs(model_dir, exist_ok=True)

    # TD3 and DDPG need action noise for exploration
    env = make_vec_env(lambda: make_env(render), n_envs=1)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=f"logs/{algo_name}/",
    )

    if algo_name in ("td3", "ddpg"):
        kwargs["action_noise"] = action_noise

    model = AlgoCls(**kwargs)

    eval_env = make_vec_env(lambda: make_env(False), n_envs=1)
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=f"{model_dir}_best/",
            log_path=f"logs/{algo_name}_eval/",
            eval_freq=5_000,
            n_eval_episodes=5,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=10_000,
            save_path=f"{model_dir}_checkpoints/",
            name_prefix=algo_name,
        ),
    ]

    print(f"\n[train.py] Starting {algo_name.upper()} for {timesteps:,} timesteps...\n")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    model.save(f"{model_dir}/final")
    print(f"\nTraining done. Model saved to {model_dir}/final.zip")


def enjoy(algo_name):
    AlgoCls = ALGOS[algo_name]
    model_path = f"models/{algo_name}_pick_place_best/best_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"No saved model at {model_path}.zip — train first.")
        return

    model = AlgoCls.load(model_path)
    env = make_env(render=True)
    obs, _ = env.reset()

    print("\nRunning trained policy — close the PyBullet window to quit.\n")
    episode_reward = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                print(f"  Episode reward: {episode_reward:.2f}")
                episode_reward = 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       default="sac", choices=["sac", "td3", "ddpg"])
    parser.add_argument("--timesteps",  type=int, default=750_000)
    parser.add_argument("--render",     action="store_true",
                        help="Open GUI during training (much slower)")
    parser.add_argument("--enjoy",      action="store_true",
                        help="Run trained policy instead of training")
    args = parser.parse_args()

    if args.enjoy:
        enjoy(args.algo)
    else:
        train(args.algo, args.timesteps, args.render)
