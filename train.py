"""
train.py  (v2 — tuned SAC hyperparameters + 500k steps default)
----------------------------------------------------------------
Usage:
    python train.py --algo sac              # default, recommended
    python train.py --algo td3
    python train.py --algo ddpg
    python train.py --algo sac --timesteps 500000
    python train.py --algo sac --render     # open GUI (slow)
    python train.py --algo sac --enjoy      # watch saved policy

Outputs:
    models/<algo>_pick_place_best/best_model.zip
    logs/diagnostics/episode_diagnostics.csv
    logs/<algo>/                             (tensorboard)
"""

import argparse
import os

import numpy as np
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

from pick_place_env import PickPlaceEnv

ALGOS = {"sac": SAC, "td3": TD3, "ddpg": DDPG}

# ── Tuned SAC hyperparameters (optimal for continuous manipulation) ─────────
SAC_KWARGS = dict(
    learning_rate  = 3e-4,
    buffer_size    = 1_000_000,
    batch_size     = 256,
    tau            = 0.005,
    gamma          = 0.99,
    train_freq     = 1,
    gradient_steps = 1,
    ent_coef       = "auto",     # auto entropy → better exploration
)

TD3_KWARGS = dict(
    learning_rate  = 1e-3,
    buffer_size    = 1_000_000,
    batch_size     = 256,
    tau            = 0.005,
    gamma          = 0.99,
    train_freq     = (1, "episode"),
    gradient_steps = -1,
    policy_delay   = 2,
)

DDPG_KWARGS = dict(
    learning_rate  = 1e-3,
    buffer_size    = 1_000_000,
    batch_size     = 256,
    tau            = 0.005,
    gamma          = 0.99,
)


def make_env(render=False):
    return PickPlaceEnv(render_mode="human" if render else None)


def train(algo_name, timesteps, render, resume=None):
    AlgoCls   = ALGOS[algo_name]
    model_dir = f"models/{algo_name}_pick_place"
    os.makedirs(model_dir, exist_ok=True)

    env       = make_vec_env(lambda: make_env(render), n_envs=1)
    n_actions = env.action_space.shape[0]

    # Action noise for TD3 / DDPG
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    if resume:
        print(f"\n[train.py] Resuming from {resume}\n")
        model = AlgoCls.load(resume, env=env)
        model.tensorboard_log = f"logs/{algo_name}/"
    else:
        algo_kwargs = {"sac": SAC_KWARGS, "td3": TD3_KWARGS, "ddpg": DDPG_KWARGS}[algo_name]

        model = AlgoCls(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=f"logs/{algo_name}/",
            **algo_kwargs,
            **({"action_noise": action_noise} if algo_name in ("td3", "ddpg") else {}),
        )

    eval_env = make_vec_env(lambda: make_env(False), n_envs=1)
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=f"{model_dir}_best/",
            log_path=f"logs/{algo_name}_eval/",
            eval_freq=10_000,
            n_eval_episodes=5,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=f"{model_dir}_checkpoints/",
            name_prefix=algo_name,
        ),
    ]

    print(f"\n[train.py] Starting {algo_name.upper()} for {timesteps:,} timesteps...\n")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    model.save(f"{model_dir}/final")
    print(f"\nTraining done. Model saved to {model_dir}/final.zip")


def enjoy(algo_name):
    AlgoCls    = ALGOS[algo_name]
    model_path = f"models/{algo_name}_pick_place_best/best_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"No saved model at {model_path}.zip — train first.")
        return

    model = AlgoCls.load(model_path)
    env   = make_env(render=True)
    obs, _ = env.reset()

    print("\nRunning trained policy — Ctrl+C to quit.\n")
    ep_reward = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            print(f"  ee_cube={info['ee_cube_dist']:.3f}  "
                  f"cube_z={info['cube_z']:.3f}  "
                  f"grasp={info.get('grasp', False)}  "
                  f"r={reward:.2f}")
            if terminated or truncated:
                print(f"  >> Episode reward: {ep_reward:.2f}\n")
                ep_reward = 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",      default="sac", choices=["sac", "td3", "ddpg"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--render",    action="store_true")
    parser.add_argument("--enjoy",     action="store_true")
    parser.add_argument("--resume",    type=str, default=None,
                        help="Path to .zip model to resume training from")
    args = parser.parse_args()

    if args.enjoy:
        enjoy(args.algo)
    else:
        train(args.algo, args.timesteps, args.render, args.resume)
