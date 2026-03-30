"""
CASID: Causal Agency Self-Improving Dense Rewards
Wrapper around Meta-World environments.

Core idea:
  1. Causal reward — max(0, corr(Δgripper, Δobj_pos)) * scale
     (proven effective in exp5/exp14; NN Granger failed at 1M steps)
  2. Self-bootstrapped OT demo reward — proximity to self-generated
     successful episode states via BallTree in feature space.
     (proven strongest in exp19 with 90% success at 1M steps)
  3. Self-improving filter — episode added to demo buffer only if
     causal score exceeds threshold (ensures buffer holds high-quality
     episodes, not just lucky ones).
"""

import gymnasium as gym
import numpy as np
from collections import deque
from sklearn.neighbors import BallTree


# ─── Hyperparameters ────────────────────────────────────────────────
CAUSAL_WINDOW   = 4       # steps of history for Granger proxy
CAUSAL_SCALE    = 0.5     # weight of causal reward
CAUSAL_THRESH   = 0.3     # min mean causal score to add ep to demo buffer

OT_SIGMA        = 0.15    # neighbourhood radius for OT reward
OT_SCALE        = 0.3     # max OT reward magnitude
OT_MAX_STATES   = 15_000  # cap demo buffer size
OT_WARMUP_EPS   = 5       # episodes before OT reward activates

# Feature normalisation (hand_obj_dist, obj_goal_dist, obj_z, gripper, near_obj, grasp)
FEAT_NORM = np.array([0.5, 0.5, 0.25, 1.0, 1.0, 1.0], dtype=np.float32)


def _extract_features(obs, info):
    """Extract 6D feature vector from Meta-World obs + info dict."""
    hand_pos = obs[0:3]
    obj_pos  = obs[6:9]
    # gripper: obs[3] is vel/gripper proxy; use action-derived state via info
    gripper  = float(info.get("grasp_success", 0.0))
    near_obj = float(info.get("near_object",   0.0))
    grasp    = float(info.get("grasp_success",  0.0))

    # We need goal pos — stored in the env, passed in via info or computed
    goal_pos = info.get("_goal_pos", np.zeros(3))
    hand_obj_dist = float(np.linalg.norm(hand_pos - obj_pos))
    obj_goal_dist = float(np.linalg.norm(obj_pos  - goal_pos))
    obj_z         = float(obj_pos[2])

    feat = np.array([hand_obj_dist, obj_goal_dist, obj_z,
                     gripper, near_obj, grasp], dtype=np.float32)
    return feat / (FEAT_NORM + 1e-8)


class SelfImprovingDemoBuffer:
    """
    Buffer of feature states from successful episodes.
    Episode is added only if its mean causal score >= CAUSAL_THRESH.
    Provides BallTree-based nearest-neighbour reward.
    """

    def __init__(self, max_states=OT_MAX_STATES, k=1, sigma=OT_SIGMA, scale=OT_SCALE):
        self.max_states = max_states
        self.k = k
        self.sigma = sigma
        self.scale = scale
        self.states: list = []
        self._tree = None
        self.n_episodes = 0

    def maybe_add_episode(self, episode_feats, episode_causal_rewards, success,
                          use_filter=True):
        """Add episode to buffer if successful (and causally scored if use_filter)."""
        if not success:
            return False
        if use_filter:
            causal_score = float(np.mean(episode_causal_rewards)) if len(episode_causal_rewards) > 0 else 0.0
            if causal_score < CAUSAL_THRESH:
                return False
        self.states.extend(episode_feats)
        # Trim to max size (keep most recent)
        if len(self.states) > self.max_states:
            self.states = self.states[-self.max_states:]
        self._tree = BallTree(np.array(self.states, dtype=np.float64))
        self.n_episodes += 1
        return True

    def reward(self, feat):
        """OT demo reward: max(0, sigma - nn_dist) / sigma * scale."""
        if self._tree is None:
            return 0.0
        k = min(self.k, len(self.states))
        dist, _ = self._tree.query(feat.reshape(1, -1).astype(np.float64), k=k)
        nn_dist = float(dist[0].mean())  # mean of k nearest distances
        return max(0.0, self.sigma - nn_dist) / self.sigma * self.scale

    def __len__(self):
        return len(self.states)


class CASIDWrapper(gym.Wrapper):
    """
    CASID reward wrapper for Meta-World environments.
    Augments the native shaped reward with:
      r_total = r_env + r_causal + r_ot_demo

    mode:
      "full"        — full CASID (causal + demo, with causal filter)  [default]
      "no_filter"   — ablation: add all successful eps, no causal filter
      "causal_only" — ablation: causal reward only, no demo reward
      "demo_only"   — ablation: demo reward only, no causal reward
      "demo_smooth" — stabilized: k=5 NN, wider sigma, stronger scale, no causal
    """

    def __init__(self, env, task_name="pick-place-v3", mode="full"):
        super().__init__(env)
        self.task_name = task_name
        self.mode = mode

        if mode == "demo_smooth":
            self.demo_buffer = SelfImprovingDemoBuffer(k=5, sigma=0.30, scale=0.5)
        else:
            self.demo_buffer = SelfImprovingDemoBuffer()

        # Causal history buffers
        self._gripper_hist = deque(maxlen=CAUSAL_WINDOW + 1)
        self._obj_hist     = deque(maxlen=CAUSAL_WINDOW + 1)

        # Episode tracking
        self._ep_feats   = []
        self._ep_causal  = []
        self._ep_success = False
        self._n_eps      = 0

        self._goal_pos   = np.zeros(3)
        self._last_info  = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._gripper_hist.clear()
        self._obj_hist.clear()
        self._ep_feats   = []
        self._ep_causal  = []
        self._ep_success = False
        self._goal_pos   = obs[36:39].copy() if obs.shape[0] >= 39 else np.zeros(3)
        self._last_info  = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store goal position (consistent throughout episode)
        if not np.allclose(obs[36:39], 0):
            self._goal_pos = obs[36:39].copy()

        # ── Causal reward ─────────────────────────────────────────
        gripper_pos = action[3]       # gripper open/close action
        obj_pos     = obs[6:9].copy()

        self._gripper_hist.append(gripper_pos)
        self._obj_hist.append(obj_pos)

        r_causal = 0.0
        if self.mode not in ("demo_only", "demo_smooth") and len(self._gripper_hist) >= CAUSAL_WINDOW:
            dg = np.diff(list(self._gripper_hist)[-CAUSAL_WINDOW:])
            do = np.diff(np.array(list(self._obj_hist)[-CAUSAL_WINDOW:]), axis=0)
            do_mag = np.linalg.norm(do, axis=1)
            if np.std(dg) > 1e-8 and np.std(do_mag) > 1e-8:
                corr = np.corrcoef(dg, do_mag)[0, 1]
                r_causal = float(max(0.0, corr)) * CAUSAL_SCALE

        # ── OT demo reward ────────────────────────────────────────
        info["_goal_pos"] = self._goal_pos
        feat = _extract_features(obs, info)
        if self.mode == "causal_only":
            r_ot = 0.0
        else:
            r_ot = self.demo_buffer.reward(feat) if self._n_eps >= OT_WARMUP_EPS else 0.0

        # ── Record episode data ───────────────────────────────────
        self._ep_feats.append(feat)
        self._ep_causal.append(r_causal)
        if info.get("success", 0.0) > 0:
            self._ep_success = True

        # ── End of episode: try to add to demo buffer ─────────────
        if terminated or truncated:
            use_filter = (self.mode not in ("no_filter", "causal_only", "demo_only", "demo_smooth"))
            added = self.demo_buffer.maybe_add_episode(
                self._ep_feats, self._ep_causal, self._ep_success,
                use_filter=use_filter
            )
            self._n_eps += 1
            info["demo_buffer_size"] = len(self.demo_buffer)
            info["demo_buffer_eps"]  = self.demo_buffer.n_episodes
            info["ep_added_to_demo"] = int(added)

        total_reward = reward + r_causal + r_ot
        info["r_env"]    = reward
        info["r_causal"] = r_causal
        info["r_ot"]     = r_ot

        return obs, total_reward, terminated, truncated, info
