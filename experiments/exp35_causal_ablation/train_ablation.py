"""
Causal ablation study: 4 methods × 2 tasks × 8 seeds

Methods:
  A — SAC baseline          (env reward, auto entropy)
  B — SAC + anneal          (env reward, annealed entropy)   ← CRITICAL ABLATION
  C — demo_smooth           (env + demo reward, auto entropy)
  D — demo_smooth + anneal  (env + demo reward, annealed entropy) ← FINAL METHOD

Usage:
  python train_ablation.py --method D --task peg-insert-side-v3 --seed 0

Mechanistic logging (paper figures):
  ent_coef_log.csv         — α per step
  success_log.csv          — per-episode success + reward
  first_success_step.txt   — first timestep above threshold
  policy_entropy_log.csv   — entropy at near-object states during eval
  qvalue_probe_log.csv     — Q(s,a) at fixed probe states per eval checkpoint
  buffer_success_log.csv   — replay buffer success fraction every 50k steps
  probe_states.npy         — saved probe states (saved at 300k or first success)
"""
import argparse
import csv
import os

import gymnasium as gym
import metaworld
import numpy as np
import torch as th
from sklearn.neighbors import BallTree
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# ──────────────────────────────────────────────
# Demo buffer (self-bootstrapped, online)
# ──────────────────────────────────────────────
OT_MAX_STATES = 50_000
OT_SIGMA = 0.30
OT_SCALE = 0.5
OT_K = 5
SUCCESS_THRESHOLD = 500.0   # raw env reward to declare success
NEAR_OBJECT_DIST = 0.05     # metres — EE→object distance for "near-object" states


class SelfImprovingDemoBuffer:
    def __init__(self):
        self.states = []
        self._tree = None

    def add(self, feat: np.ndarray):
        self.states.append(feat.astype(np.float64))
        if len(self.states) > OT_MAX_STATES:
            self.states.pop(0)
        self._tree = None

    def _build_tree(self):
        if self._tree is None and len(self.states) > OT_K:
            self._tree = BallTree(np.array(self.states))

    def reward(self, feat: np.ndarray) -> float:
        self._build_tree()
        if self._tree is None:
            return 0.0
        k = min(OT_K, len(self.states))
        dist, _ = self._tree.query(feat.reshape(1, -1).astype(np.float64), k=k)
        nn_dist = float(dist[0].mean())
        return max(0.0, OT_SIGMA - nn_dist) / OT_SIGMA * OT_SCALE


# ──────────────────────────────────────────────
# Env wrapper (methods C and D)
# ──────────────────────────────────────────────
class DemoSmoothWrapper(gym.Wrapper):
    """Adds self-bootstrapped k-NN demo reward on top of env reward."""

    def __init__(self, env):
        super().__init__(env)
        self.demo_buffer = SelfImprovingDemoBuffer()
        self._episode_reward = 0.0

    @staticmethod
    def _feat(obs: np.ndarray) -> np.ndarray:
        return obs[:6].copy()  # EE pos (0:3) + object pos (3:6)

    @staticmethod
    def _ee_object_dist(obs: np.ndarray) -> float:
        return float(np.linalg.norm(obs[:3] - obs[3:6]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        feat = self._feat(obs)
        demo_r = self.demo_buffer.reward(feat)
        shaped_reward = reward + demo_r
        self._episode_reward += reward

        success = info.get("success", False) or (self._episode_reward > SUCCESS_THRESHOLD)
        if success:
            self.demo_buffer.add(feat)

        info["demo_reward"] = demo_r
        info["success"] = success
        info["ee_object_dist"] = self._ee_object_dist(obs)
        return obs, shaped_reward, terminated, truncated, info


class BaseEnvWithDist(gym.Wrapper):
    """Thin wrapper for A/B that adds ee_object_dist to info."""

    @staticmethod
    def _ee_object_dist(obs: np.ndarray) -> float:
        return float(np.linalg.norm(obs[:3] - obs[3:6]))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["ee_object_dist"] = self._ee_object_dist(obs)
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────
# Entropy annealing helper
# ──────────────────────────────────────────────
def compute_ent(step: int, ent_start=0.1, ent_end=0.005,
                anneal_start=100_000, anneal_end=500_000) -> float:
    if step < anneal_start:
        return ent_start
    if step >= anneal_end:
        return ent_end
    progress = (step - anneal_start) / (anneal_end - anneal_start)
    return ent_start - progress * (ent_start - ent_end)


# ──────────────────────────────────────────────
# Mechanistic logging callback (training-time)
# ──────────────────────────────────────────────
class MechanisticLoggingCallback(BaseCallback):
    """
    Handles four things each training step:
      1. Entropy annealing (methods B, D)
      2. ent_coef_log.csv
      3. success_log.csv + first_success_step.txt
      4. buffer_success_log.csv (every 50k steps)
      5. Collects near-object probe states → saved at 300k or first success
    """

    PROBE_SAVE_STEPS = 300_000
    BUFFER_CHECK_EVERY = 50_000
    ENT_LOG_EVERY = 1_000
    MAX_PROBES = 20

    def __init__(self, logdir: str, anneal: bool):
        super().__init__(verbose=0)
        self.anneal = anneal
        self.logdir = logdir

        self.first_success_step = None
        self._current_ep_reward = 0.0
        self._episode_count = 0
        self._probe_states_saved = False
        self._probe_states: list = []       # (obs, action) tuples
        self._last_buffer_check = 0

        os.makedirs(logdir, exist_ok=True)

        self._ent_f = open(os.path.join(logdir, "ent_coef_log.csv"), "w", newline="")
        self._ent_w = csv.writer(self._ent_f)
        self._ent_w.writerow(["step", "ent_coef"])

        self._suc_f = open(os.path.join(logdir, "success_log.csv"), "w", newline="")
        self._suc_w = csv.writer(self._suc_f)
        self._suc_w.writerow(["step", "episode", "success", "ep_reward"])

        self._buf_f = open(os.path.join(logdir, "buffer_success_log.csv"), "w", newline="")
        self._buf_w = csv.writer(self._buf_f)
        self._buf_w.writerow(["step", "buffer_success_fraction"])

    # ── entropy override ───────────────────────
    def _apply_entropy(self, step: int):
        ent_val = compute_ent(step)
        # Update both the float and the tensor used in the actor loss
        self.model.ent_coef = ent_val
        self.model.ent_coef_tensor = th.tensor(
            ent_val, dtype=th.float32, device=self.model.device
        )

    def _current_ent_coef(self) -> float:
        if self.anneal:
            return compute_ent(self.num_timesteps)
        try:
            return float(self.model.ent_coef_tensor.detach().cpu())
        except Exception:
            v = self.model.ent_coef
            return float(v) if isinstance(v, float) else -1.0

    # ── probe state collection ─────────────────
    def _try_collect_probe(self, obs_arr, action_arr, info_dict):
        if self._probe_states_saved or len(self._probe_states) >= self.MAX_PROBES:
            return
        dist = info_dict.get("ee_object_dist", 999.0)
        if dist < NEAR_OBJECT_DIST:
            self._probe_states.append((obs_arr.copy(), action_arr.copy()))

    def _save_probes(self):
        if self._probe_states_saved or not self._probe_states:
            return
        probes = {
            "obs": np.array([p[0] for p in self._probe_states]),
            "act": np.array([p[1] for p in self._probe_states]),
        }
        np.save(os.path.join(self.logdir, "probe_states.npy"), probes, allow_pickle=True)
        self._probe_states_saved = True
        print(f"[PROBES] Saved {len(self._probe_states)} probe states at step {self.num_timesteps}", flush=True)

    # ── replay buffer success fraction ─────────
    def _check_buffer(self, step: int):
        buf = self.model.replay_buffer
        if buf.n_envs == 0 or buf.size() < 100:
            return
        sample = buf.sample(min(1000, buf.size()))
        rewards = sample.rewards.cpu().numpy().flatten()
        frac = float((rewards > SUCCESS_THRESHOLD / 500).mean())  # normalize: shaped reward peaks ~scale
        self._buf_w.writerow([step, round(frac, 4)])
        self._buf_f.flush()

    # ── main step ─────────────────────────────
    def _on_step(self) -> bool:
        step = self.num_timesteps

        # 1. Entropy annealing
        if self.anneal:
            self._apply_entropy(step)

        # 2. Log ent_coef
        if step % self.ENT_LOG_EVERY == 0:
            self._ent_w.writerow([step, round(self._current_ent_coef(), 6)])

        # 3. Episode success tracking
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [{}])
        obs = self.locals.get("new_obs", None)
        actions = self.locals.get("actions", None)

        for i, (r, done, info) in enumerate(zip(rewards, dones, infos)):
            self._current_ep_reward += float(r)

            # Collect probe states during training
            if obs is not None and actions is not None:
                self._try_collect_probe(obs[i], actions[i], info)

            if done:
                success = info.get("success", False) or (self._current_ep_reward > SUCCESS_THRESHOLD)
                self._suc_w.writerow([step, self._episode_count, int(success), round(self._current_ep_reward, 2)])

                if success and self.first_success_step is None:
                    self.first_success_step = step
                    print(f"[FIRST SUCCESS] step={step:,}", flush=True)
                    # Save probes immediately on first success
                    self._save_probes()

                self._episode_count += 1
                self._current_ep_reward = 0.0

        # 4. Save probes at 300k if not yet saved
        if step >= self.PROBE_SAVE_STEPS and not self._probe_states_saved:
            self._save_probes()

        # 5. Buffer success fraction every 50k steps
        if step - self._last_buffer_check >= self.BUFFER_CHECK_EVERY:
            self._check_buffer(step)
            self._last_buffer_check = step

        return True

    def _on_training_end(self):
        for f in [self._ent_f, self._suc_f, self._buf_f]:
            f.flush(); f.close()
        with open(os.path.join(self.logdir, "first_success_step.txt"), "w") as f:
            f.write(str(self.first_success_step) if self.first_success_step else "never")
        print(f"[DONE] first_success={self.first_success_step}", flush=True)


# ──────────────────────────────────────────────
# Custom eval callback with mechanistic metrics
# ──────────────────────────────────────────────
class MechanisticEvalCallback(BaseCallback):
    """
    Runs its own eval loop every eval_freq steps and logs:
      - Mean/std eval reward  → evaluations.npz (compatible with SB3)
      - Policy entropy at near-object states → policy_entropy_log.csv
      - Q-value at probe states → qvalue_probe_log.csv
    """

    def __init__(self, eval_env, logdir: str, eval_freq: int = 10_000,
                 n_eval_episodes: int = 20, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.logdir = logdir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval = 0

        # npz-compatible storage
        self._timesteps = []
        self._results = []      # shape: (n_evals, n_eval_episodes)
        self._ep_lengths = []

        os.makedirs(os.path.join(logdir, "eval"), exist_ok=True)

        self._ent_f = open(os.path.join(logdir, "policy_entropy_log.csv"), "w", newline="")
        self._ent_w = csv.writer(self._ent_f)
        self._ent_w.writerow(["step", "mean_near_object_entropy", "n_near_object_states"])

        self._qf = open(os.path.join(logdir, "qvalue_probe_log.csv"), "w", newline="")
        self._qw = csv.writer(self._qf)
        self._qw.writerow(["step", "mean_q", "std_q", "n_probes"])

    def _load_probes(self):
        """Load probe states if available."""
        probe_path = os.path.join(self.logdir, "probe_states.npy")
        if os.path.exists(probe_path):
            return np.load(probe_path, allow_pickle=True).item()
        return None

    def _eval_policy_entropy_and_qvalues(self, step: int):
        """Run n_eval_episodes, collect near-object entropy and Q-value probes."""
        policy = self.model.policy
        policy.set_training_mode(False)

        near_obj_entropies = []
        ep_rewards = []
        ep_lengths = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0

            while not done:
                obs_tensor = th.tensor(obs, dtype=th.float32, device=self.model.device).unsqueeze(0)

                with th.no_grad():
                    # Policy entropy at near-object states
                    # SquashedDiagGaussianDistribution.entropy() returns None (squashing breaks it)
                    # Use underlying Gaussian entropy: sum over action dims, mean over batch
                    mean_a, log_std, _ = policy.actor.get_action_dist_params(obs_tensor)
                    gauss_dist = policy.actor.action_dist.proba_distribution(mean_a, log_std)
                    entropy = gauss_dist.distribution.entropy().sum(dim=-1).mean().item()

                    ee_dist = float(np.linalg.norm(obs[:3] - obs[3:6]))
                    if ee_dist < NEAR_OBJECT_DIST:
                        near_obj_entropies.append(entropy)

                    action, _ = policy.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = terminated or truncated

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)

        policy.set_training_mode(True)

        # Log near-object entropy
        mean_entropy = float(np.mean(near_obj_entropies)) if near_obj_entropies else -1.0
        self._ent_w.writerow([step, round(mean_entropy, 6), len(near_obj_entropies)])
        self._ent_f.flush()

        # Q-value probes
        probes = self._load_probes()
        if probes is not None:
            obs_np = probes["obs"]
            act_np = probes["act"]
            obs_t = th.tensor(obs_np, dtype=th.float32, device=self.model.device)
            act_t = th.tensor(act_np, dtype=th.float32, device=self.model.device)
            with th.no_grad():
                q1, q2 = self.model.critic(obs_t, act_t)
                q_vals = th.min(q1, q2).cpu().numpy().flatten()
            self._qw.writerow([step, round(float(q_vals.mean()), 4),
                               round(float(q_vals.std()), 4), len(q_vals)])
            self._qf.flush()

        return np.array(ep_rewards), np.array(ep_lengths)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last_eval < self.eval_freq:
            return True
        self._last_eval = step

        ep_rewards, ep_lengths = self._eval_policy_entropy_and_qvalues(step)

        self._timesteps.append(step)
        self._results.append(ep_rewards)
        self._ep_lengths.append(ep_lengths)

        mean_r = float(ep_rewards.mean())
        std_r = float(ep_rewards.std())

        if self.verbose:
            print(f"[EVAL] step={step:,}  mean={mean_r:.1f} ± {std_r:.1f}", flush=True)

        # Save evaluations.npz (SB3-compatible format)
        np.savez(
            os.path.join(self.logdir, "eval", "evaluations.npz"),
            timesteps=np.array(self._timesteps),
            results=np.array(self._results),
            ep_lengths=np.array(self._ep_lengths),
        )
        return True

    def _on_training_end(self):
        self._ent_f.flush(); self._ent_f.close()
        self._qf.flush(); self._qf.close()


# ──────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────
def make_env(task_name: str, method: str, seed: int):
    ml1 = metaworld.ML1(task_name, seed=seed)
    env_cls = ml1.train_classes[task_name]
    task = ml1.train_tasks[0]
    raw_env = env_cls()
    raw_env.set_task(task)

    if method in ("C", "D"):
        env = DemoSmoothWrapper(raw_env)
    else:
        env = BaseEnvWithDist(raw_env)

    return Monitor(env)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()

    use_anneal = args.method in ("B", "D")
    run_id = f"{args.task}__method{args.method}__seed{args.seed}"
    logdir = os.path.join(args.logdir, run_id)
    os.makedirs(logdir, exist_ok=True)

    print(f"[START] method={args.method} task={args.task} seed={args.seed} anneal={use_anneal}", flush=True)

    train_env = make_env(args.task, args.method, args.seed)
    eval_env = make_env(args.task, args.method, args.seed)

    # SAC — locked hyperparameters
    # For annealed runs: start at fixed 0.1 (callback takes over immediately)
    # For auto runs: 'auto' lets SB3 tune α normally
    model = SAC(
        "MlpPolicy",
        train_env,
        batch_size=256,
        buffer_size=1_000_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        ent_coef=0.1 if use_anneal else "auto",
        target_entropy="auto",
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,
        device="cpu",
    )

    logging_cb = MechanisticLoggingCallback(logdir=logdir, anneal=use_anneal)
    eval_cb = MechanisticEvalCallback(
        eval_env=eval_env,
        logdir=logdir,
        eval_freq=10_000,
        n_eval_episodes=20,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=[logging_cb, eval_cb],
        progress_bar=False,
    )

    model.save(os.path.join(logdir, "final_model"))
    train_env.close()
    eval_env.close()
    print(f"[COMPLETE] {run_id}", flush=True)


if __name__ == "__main__":
    main()
