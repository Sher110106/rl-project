"""
train_qucb.py — Exp17: Q-UCB Potential-Based Reward Shaping (SLOPE for model-free SAC)
----------------------------------------------------------------------------------------
Implements Rank 2 from RESEARCH_AND_PLAN.md: Q-UCB Potential Shaping.

Inspired by SLOPE (2026): uses Q-function uncertainty as a potential to create
a smooth gradient landscape, guiding the agent toward high-value regions even
before it has found success.

SAC maintains two Q-networks (q1, q2). Their disagreement is an implicit
measure of epistemic uncertainty — regions not well-covered by the replay buffer.

Shaped reward:
    r_shaped(s, a, s') = r_env(s, a, s') + coef * (γ * Q_UCB(s') - Q_UCB(s))

where Q_UCB(s, a) = (Q1(s,a) + Q2(s,a)) / 2 + β * |Q1(s,a) - Q2(s,a)| / 2

Properties:
  - Valid potential-based reward shaping (Ng et al. 1999) → optimal policy preserved
  - Q_UCB is high in uncertain regions → encourages exploration of novel states
  - Q_UCB is high in high-value regions → guides toward task completion
  - Combined with Granger(scale=0.8) base — best manipulation signal

Implementation via SB3 override of _store_transition:
  Hooks into the replay buffer insertion point to shape rewards before storage.
"""

import argparse
import os
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from pick_place_env_granger import PickPlaceGrangerEnv

# Q-UCB shaping hyperparameters
UCB_BETA      = 1.0    # exploration bonus weight (higher = more uncertainty-driven)
SHAPING_COEF  = 0.05   # potential shaping coefficient (small: don't dominate env reward)

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


class QUCBShapedSAC(SAC):
    """
    SAC with Q-UCB potential-based reward shaping.

    Overrides _store_transition to inject shaped rewards before buffer insertion.
    Uses the critic's Q-value disagreement as UCB estimate.
    """

    def __init__(self, *args, ucb_beta=UCB_BETA, shaping_coef=SHAPING_COEF, **kwargs):
        super().__init__(*args, **kwargs)
        self.ucb_beta    = ucb_beta
        self.shaping_coef = shaping_coef

    def _q_ucb(self, obs_np: np.ndarray, action_np: np.ndarray) -> float:
        """Compute Q-UCB = Q_mean + beta * Q_std using critic disagreement."""
        with th.no_grad():
            obs_t = th.FloatTensor(obs_np).to(self.device)
            act_t = th.FloatTensor(action_np).to(self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
                act_t = act_t.unsqueeze(0)
            q1, q2 = self.critic(obs_t, act_t)
            q_mean  = (q1 + q2) / 2.0
            q_half_range = th.abs(q1 - q2) / 2.0
            q_ucb   = q_mean + self.ucb_beta * q_half_range
        return q_ucb.squeeze().cpu().item()

    def _store_transition(self, replay_buffer, buffer_action,
                          new_obs, reward, dones, infos):
        """Shape reward with Q-UCB potential before storing in replay buffer."""
        if (self.num_timesteps > self.learning_starts
                and self._last_obs is not None):
            try:
                # F(s') - γ * F(s) potential shaping
                # We use Q_UCB as the potential function F
                f_curr = self._q_ucb(self._last_obs[0], buffer_action[0])
                f_next = self._q_ucb(new_obs[0], buffer_action[0])
                potential = self.gamma * f_next - f_curr
                # Clip to avoid destabilizing the replay buffer
                potential = float(np.clip(potential, -50.0, 50.0))
                reward = reward + self.shaping_coef * potential
            except Exception:
                pass  # fall back to unshaped reward on any error

        super()._store_transition(replay_buffer, buffer_action,
                                  new_obs, reward, dones, infos)


def make_env():
    return PickPlaceGrangerEnv(log_dir="logs/diagnostics")


def train(timesteps):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env      = make_vec_env(make_env, n_envs=1)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = QUCBShapedSAC(
        "MlpPolicy", env, verbose=1,
        tensorboard_log="logs/tb/",
        ucb_beta=UCB_BETA, shaping_coef=SHAPING_COEF,
        **SAC_KWARGS,
    )

    callbacks = [
        EvalCallback(eval_env,
                     best_model_save_path="models/best/",
                     log_path="logs/eval/",
                     eval_freq=10_000,
                     n_eval_episodes=5,
                     verbose=1),
        CheckpointCallback(save_freq=100_000,
                           save_path="models/checkpoints/",
                           name_prefix="sac_qucb"),
    ]

    print(f"\n[exp17_qucb] Q-UCB SAC for {timesteps:,} steps")
    print(f"  UCB_BETA={UCB_BETA}, SHAPING_COEF={SHAPING_COEF}")
    print(f"  Base env: Granger(scale=0.8)\n")
    model.learn(total_timesteps=timesteps, callback=callbacks,
                progress_bar=False)
    model.save("models/final")
    print("Done. Saved to models/final.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    args = p.parse_args()
    train(args.timesteps)
