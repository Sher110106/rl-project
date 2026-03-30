"""
train_true_granger_qucb.py — Exp18: True Granger + Q-UCB Potential Shaping
----------------------------------------------------------------------------
Combines both Rank 1 (proper NN Granger) and Rank 2 (Q-UCB shaping).

True Granger handles: grasp-phase causal signal (what to do when near cube)
Q-UCB handles:        exploration landscape (where to go in state space)

These address complementary aspects:
- Granger = dense manipulation quality signal (local, contact-phase)
- Q-UCB   = global exploration bonus toward high-value states
"""
import argparse, os
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from pick_place_env_true_granger import PickPlaceTrueGrangerEnv

UCB_BETA     = 1.0
SHAPING_COEF = 0.05

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
    def __init__(self, *args, ucb_beta=UCB_BETA, shaping_coef=SHAPING_COEF, **kwargs):
        super().__init__(*args, **kwargs)
        self.ucb_beta     = ucb_beta
        self.shaping_coef = shaping_coef

    def _q_ucb(self, obs_np, action_np):
        with th.no_grad():
            obs_t = th.FloatTensor(obs_np).to(self.device)
            act_t = th.FloatTensor(action_np).to(self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
                act_t = act_t.unsqueeze(0)
            q1, q2 = self.critic(obs_t, act_t)
            q_ucb = (q1 + q2) / 2.0 + self.ucb_beta * th.abs(q1 - q2) / 2.0
        return q_ucb.squeeze().cpu().item()

    def _store_transition(self, replay_buffer, buffer_action,
                          new_obs, reward, dones, infos):
        if self.num_timesteps > self.learning_starts and self._last_obs is not None:
            try:
                f_curr = self._q_ucb(self._last_obs[0], buffer_action[0])
                f_next = self._q_ucb(new_obs[0],        buffer_action[0])
                potential = float(np.clip(self.gamma * f_next - f_curr, -50.0, 50.0))
                reward    = reward + self.shaping_coef * potential
            except Exception:
                pass
        super()._store_transition(replay_buffer, buffer_action,
                                  new_obs, reward, dones, infos)


def make_env():
    return PickPlaceTrueGrangerEnv(log_dir="logs/diagnostics")

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
                           name_prefix="sac_tg_qucb"),
    ]

    print(f"\n[exp18_true_granger_qucb] SAC for {timesteps:,} steps")
    print(f"  True NN Granger + Q-UCB(beta={UCB_BETA}, coef={SHAPING_COEF})\n")
    model.learn(total_timesteps=timesteps, callback=callbacks,
                progress_bar=False)
    model.save("models/final")
    print("Done. Saved to models/final.zip")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    args = p.parse_args()
    train(args.timesteps)
