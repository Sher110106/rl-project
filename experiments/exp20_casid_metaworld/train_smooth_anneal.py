"""
Demo_smooth + entropy annealing: the stabilization hypothesis.

Key idea: SAC's auto-tuned entropy keeps exploring AFTER finding the solution,
causing forgetting. We fix this by:
  1. Starting with high entropy (ent_coef=0.1) for broad exploration
  2. Annealing to low entropy (ent_coef=0.005) to lock in discovered behaviors

Combined with demo_smooth (k=5 NN, wider sigma, stronger scale) for a
smoother reward signal that resists policy drift.

Run: python train_smooth_anneal.py --task peg-insert-side-v3 --seed 0 --steps 1000000
"""

import argparse
import os
import numpy as np
import metaworld
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from casid_env import CASIDWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--task",       default="peg-insert-side-v3")
parser.add_argument("--seed",       type=int, default=0)
parser.add_argument("--steps",      type=int, default=1_000_000)
parser.add_argument("--logdir",     default="logs")
parser.add_argument("--ent_start",  type=float, default=0.1)
parser.add_argument("--ent_end",    type=float, default=0.005)
parser.add_argument("--anneal_start", type=int, default=100_000,
                    help="Step at which annealing begins")
parser.add_argument("--anneal_end",   type=int, default=500_000,
                    help="Step at which entropy reaches ent_end")
args = parser.parse_args()

LOG_DIR  = os.path.join(args.logdir, f"{args.task}_seed{args.seed}")
EVAL_DIR = os.path.join(LOG_DIR, "eval")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"[SmoothAnneal] task={args.task} seed={args.seed}")
print(f"[SmoothAnneal] entropy: {args.ent_start} → {args.ent_end} over steps {args.anneal_start}–{args.anneal_end}")


class EntropyAnnealCallback(BaseCallback):
    """Linearly anneal SAC's ent_coef from start to end over a step range."""

    def __init__(self, ent_start, ent_end, anneal_start, anneal_end, verbose=0):
        super().__init__(verbose)
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step < self.anneal_start:
            ent = self.ent_start
        elif step >= self.anneal_end:
            ent = self.ent_end
        else:
            frac = (step - self.anneal_start) / (self.anneal_end - self.anneal_start)
            ent = self.ent_start + frac * (self.ent_end - self.ent_start)

        # Override SAC's entropy coefficient (bypass auto-tuning)
        self.model.ent_coef = ent
        # Also update the log_ent_coef tensor so logging is correct
        self.model.log_ent_coef = None  # disable auto-tuning path

        if step % 50_000 == 0:
            print(f"  [EntropyAnneal] step={step:,} ent_coef={ent:.4f}")
        return True


def make_env(task_name, seed=0):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[seed % len(ml1.train_tasks)])
    env = CASIDWrapper(env, task_name=task_name, mode="demo_smooth")
    return Monitor(env)


def make_eval_env(task_name, seed=0):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[seed % len(ml1.train_tasks)])
    return Monitor(env)


train_env = make_env(args.task, args.seed)
eval_env = make_eval_env(args.task, args.seed)

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef=args.ent_start,  # fixed start, not "auto"
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=os.path.join(LOG_DIR, "tb"),
    seed=args.seed,
    verbose=1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=EVAL_DIR,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

anneal_cb = EntropyAnnealCallback(
    ent_start=args.ent_start,
    ent_end=args.ent_end,
    anneal_start=args.anneal_start,
    anneal_end=args.anneal_end,
)

model.learn(
    total_timesteps=args.steps,
    callback=[eval_cb, anneal_cb],
    progress_bar=True,
)
model.save(os.path.join(LOG_DIR, "final_model"))
print(f"[SmoothAnneal] Done. Final ent_coef={model.ent_coef}")
