"""
train_ot.py — Exp19: Self-Bootstrapped OT Demo Reward
"""
import argparse, os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from pick_place_env_ot import PickPlaceOTEnv

SAC_KWARGS = dict(
    learning_rate  = 3e-4,
    buffer_size    = 1_000_000,
    batch_size     = 256,
    tau            = 0.005,
    gamma          = 0.99,
    train_freq     = 1,
    gradient_steps = 1,
    ent_coef       = "auto",
)

def make_env():
    return PickPlaceOTEnv(log_dir="logs/diagnostics", demo_path="ot_demos.npy")

def train(timesteps):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env      = make_vec_env(make_env, n_envs=1)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = SAC("MlpPolicy", env, verbose=1,
                tensorboard_log="logs/tb/", **SAC_KWARGS)

    callbacks = [
        EvalCallback(eval_env,
                     best_model_save_path="models/best/",
                     log_path="logs/eval/",
                     eval_freq=10_000,
                     n_eval_episodes=5,
                     verbose=1),
        CheckpointCallback(save_freq=100_000,
                           save_path="models/checkpoints/",
                           name_prefix="sac_ot"),
    ]

    print(f"\n[exp19_ot_demo] SAC for {timesteps:,} steps")
    print("  OT demo reward (NN proximity to successful exp5 states)")
    print("  + Granger(scale=0.8) + dense base\n")
    model.learn(total_timesteps=timesteps, callback=callbacks,
                progress_bar=False)
    model.save("models/final")
    print("Done. Saved to models/final.zip")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    args = p.parse_args()
    train(args.timesteps)
