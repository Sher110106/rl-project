"""
Exp20: CASID on Meta-World pick-place-v3
Run: python train_casid.py [--task pick-place-v3] [--seed 0] [--steps 1000000]
"""

import argparse
import os
import sys
import numpy as np
import metaworld
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from casid_env import CASIDWrapper

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task",  default="pick-place-v3")
parser.add_argument("--seed",  type=int, default=0)
parser.add_argument("--steps", type=int, default=1_000_000)
parser.add_argument("--logdir", default="logs")
parser.add_argument("--mode",  default="full",
                    choices=["full", "no_filter", "causal_only", "demo_only", "demo_smooth"])
args = parser.parse_args()

LOG_DIR  = os.path.join(args.logdir, f"{args.task}_seed{args.seed}")
EVAL_DIR = os.path.join(LOG_DIR, "eval")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"[CASID] task={args.task} seed={args.seed} steps={args.steps} mode={args.mode}")
print(f"[CASID] logdir={LOG_DIR}")

# ─── env factory ─────────────────────────────────────────────────────
def make_env(task_name, seed=0):
    ml1  = metaworld.ML1(task_name)
    env  = ml1.train_classes[task_name]()
    task = ml1.train_tasks[seed % len(ml1.train_tasks)]
    env.set_task(task)
    env = CASIDWrapper(env, task_name=task_name, mode=args.mode)
    env = Monitor(env)
    return env

def make_eval_env(task_name, seed=0):
    ml1  = metaworld.ML1(task_name)
    env  = ml1.train_classes[task_name]()
    task = ml1.train_tasks[seed % len(ml1.train_tasks)]
    env.set_task(task)
    # Eval env without CASID shaping — measures true task performance
    env = Monitor(env)
    return env

# ─── build envs ──────────────────────────────────────────────────────
train_env = make_env(args.task, args.seed)
eval_env  = make_eval_env(args.task, args.seed)

# ─── SAC model ───────────────────────────────────────────────────────
model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=os.path.join(LOG_DIR, "tb"),
    seed=args.seed,
    verbose=1,
)

# ─── eval callback ───────────────────────────────────────────────────
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=EVAL_DIR,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

# ─── train ───────────────────────────────────────────────────────────
model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=True)

model.save(os.path.join(LOG_DIR, "final_model"))
print("[CASID] Training complete.")
print(f"[CASID] Demo buffer episodes: {train_env.unwrapped.demo_buffer.n_episodes}")
print(f"[CASID] Demo buffer size:     {len(train_env.unwrapped.demo_buffer)}")
