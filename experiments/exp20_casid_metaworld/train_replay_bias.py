"""
Replay-biased SAC: duplicate successful transitions in the replay buffer.
Addresses the "discover then forget" failure mode by biasing replay toward
successful experiences.

Run: python train_replay_bias.py [--task pick-place-v3] [--seed 0] [--steps 1000000]
                                 [--mode demo_only] [--replay_k 10]
"""

import argparse
import os
import numpy as np
import metaworld
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from casid_env import CASIDWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--task",     default="pick-place-v3")
parser.add_argument("--seed",     type=int, default=0)
parser.add_argument("--steps",    type=int, default=1_000_000)
parser.add_argument("--logdir",   default="logs")
parser.add_argument("--mode",     default="demo_only",
                    choices=["full", "no_filter", "causal_only", "demo_only", "demo_smooth"])
parser.add_argument("--replay_k", type=int, default=10,
                    help="Number of extra copies of successful transitions")
args = parser.parse_args()

LOG_DIR  = os.path.join(args.logdir, f"{args.task}_seed{args.seed}")
EVAL_DIR = os.path.join(LOG_DIR, "eval")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"[ReplayBias] task={args.task} seed={args.seed} mode={args.mode} replay_k={args.replay_k}")


class ReplayBiasedSAC(SAC):
    """SAC with success-biased replay: duplicate transitions where success=True."""

    def __init__(self, *a, replay_k=10, **kw):
        super().__init__(*a, **kw)
        self.replay_k = replay_k
        self._total_duplicated = 0

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        # Normal store
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

        # Check if any env in the vectorized batch has success
        for info in infos:
            if info.get("success", 0) > 0:
                # Duplicate this successful transition
                for _ in range(self.replay_k):
                    super()._store_transition(
                        replay_buffer, buffer_action, new_obs, reward, dones, infos
                    )
                self._total_duplicated += self.replay_k
                break  # single env, so break after first


def make_env(task_name, seed=0):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[seed % len(ml1.train_tasks)])
    env = CASIDWrapper(env, task_name=task_name, mode=args.mode)
    return Monitor(env)


def make_eval_env(task_name, seed=0):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[seed % len(ml1.train_tasks)])
    return Monitor(env)


train_env = make_env(args.task, args.seed)
eval_env = make_eval_env(args.task, args.seed)

model = ReplayBiasedSAC(
    "MlpPolicy",
    train_env,
    replay_k=args.replay_k,
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

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=EVAL_DIR,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=True)
model.save(os.path.join(LOG_DIR, "final_model"))
print(f"[ReplayBias] Training complete. Duplicated transitions: {model._total_duplicated}")
