"""
Experiment 16: True Neural-Network Granger-Causal Reward
---------------------------------------------------------
Fixes the approximation in Exp5-14 which used cross-correlation as a proxy.

TRUE Granger causality test (Granger 1969):
  Variable A Granger-causes variable B if knowing A's history reduces
  prediction error for B's future beyond using B's history alone.

  Formally: Var(B_{t+1} | B_t) > Var(B_{t+1} | B_t, A_t)
  i.e., the causal model (with A) beats the baseline (without A).

Implementation:
  - Causal model:   MLP(gripper_history[8] + cube_pos[3]) → Δcube_pos[3]
  - Baseline model: MLP(cube_pos[3]) → Δcube_pos[3]
  - Both trained online via SGD at every step.
  - Granger reward = max(0, MSE_baseline - MSE_causal) * SCALE

  When the gripper IS causally influencing the cube:
    Causal model predicts cube motion much better than baseline → high reward
  When the gripper is NOT causing cube motion (agent just nearby):
    Both models perform similarly → near-zero reward

This is semantically correct: reward measures actual manipulation agency,
not just spatial correlation or proximity.

Combined with:
  - Dense distance shaping (reach + tray pull)
  - Dense gripper proximity shaping
  - One-time bonuses (anti-gaming)
  - CAUSAL_SCALE = 2.0 (tuned for MSE-difference scale, not corr scale)
"""

import csv
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

ROBOT_URDF    = "franka_panda/panda.urdf"
ARM_JOINTS    = list(range(7))
FINGER_JOINTS = [9, 10]
FINGER_LINKS  = [9, 10]
EE_LINK       = 11
FINGER_OPEN   = 0.04
FINGER_CLOSED = 0.00
REST_POSE     = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

PLACE_THRESHOLD   = 0.05
LIFT_THRESHOLD    = 0.05
GRIPPER_PROXIMITY = 0.08
MAX_STEPS         = 500
DELTA_CLIP        = 0.05
N_SUBSTEPS        = 10
WORKSPACE = {
    "low":  np.array([0.25, -0.55, 0.01]),
    "high": np.array([0.80,  0.55, 0.70]),
}

# True Granger config
GRANGER_WINDOW    = 8      # steps of gripper history
GRANGER_HIDDEN    = 32     # NN hidden units
GRANGER_LR        = 3e-3   # online SGD learning rate
GRANGER_SCALE     = 2.0    # reward scale (MSE difference units)
GRANGER_PROXIMITY = 0.15   # only active within this EE-cube distance
GRANGER_WARMUP    = 16     # min steps before computing reward


class NeuralGrangerCausalReward:
    """
    True Granger causality via paired neural networks.

    Trains online: at each step, both models predict Δcube_pos,
    then the reward is how much more the causal model (with gripper info) knows.
    """

    def __init__(self, window=GRANGER_WINDOW, hidden=GRANGER_HIDDEN, lr=GRANGER_LR):
        self.window = window
        self.scale  = GRANGER_SCALE

        # Causal model: gripper_history[window] + cube_pos[3] → Δcube_pos[3]
        self.causal_net = nn.Sequential(
            nn.Linear(window + 3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 3),
        )

        # Baseline model: cube_pos[3] → Δcube_pos[3]  (no gripper info)
        self.baseline_net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),
        )

        self.causal_opt   = optim.Adam(self.causal_net.parameters(),   lr=lr)
        self.baseline_opt = optim.Adam(self.baseline_net.parameters(), lr=lr)
        self.loss_fn      = nn.MSELoss()

        self._gripper_hist = deque(maxlen=window)
        self._cube_hist    = deque(maxlen=window + 2)
        self._step         = 0

    def reset(self):
        self._gripper_hist.clear()
        self._cube_hist.clear()
        self._step = 0

    def update_and_compute(self, gripper_state: float,
                           cube_pos: np.ndarray,
                           ee_cube_dist: float) -> float:
        """Update both NNs with the new transition and return Granger reward."""
        self._gripper_hist.append(float(gripper_state))
        self._cube_hist.append(cube_pos.copy())
        self._step += 1

        # Only compute when close and have enough history
        if (ee_cube_dist > GRANGER_PROXIMITY
                or self._step < GRANGER_WARMUP
                or len(self._gripper_hist) < self.window
                or len(self._cube_hist) < self.window + 1):
            return 0.0

        # Build tensors
        g_hist = torch.FloatTensor(list(self._gripper_hist))         # (window,)
        cube_h = np.array(list(self._cube_hist))
        c_prev = torch.FloatTensor(cube_h[-2])                       # (3,)
        c_curr = torch.FloatTensor(cube_h[-1])                       # (3,)
        delta  = c_curr - c_prev                                     # target (3,)

        # --- Train causal model ---
        self.causal_opt.zero_grad()
        causal_in  = torch.cat([g_hist, c_prev])                     # (window+3,)
        causal_out = self.causal_net(causal_in)
        c_loss = self.loss_fn(causal_out, delta)
        c_loss.backward()
        self.causal_opt.step()

        # --- Train baseline model ---
        self.baseline_opt.zero_grad()
        baseline_out = self.baseline_net(c_prev)
        b_loss = self.loss_fn(baseline_out, delta)
        b_loss.backward()
        self.baseline_opt.step()

        # --- Compute Granger reward (no gradient) ---
        with torch.no_grad():
            c_mse = self.loss_fn(self.causal_net(causal_in), delta).item()
            b_mse = self.loss_fn(self.baseline_net(c_prev),  delta).item()

        # Positive only when causal model beats baseline
        granger_r = max(0.0, b_mse - c_mse) * self.scale

        # Clip to prevent reward spikes on large MSE differences
        return float(min(granger_r, 1.0))


class PickPlaceTrueGrangerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, log_dir="logs/diagnostics"):
        super().__init__()
        self.render_mode = render_mode
        self.log_dir     = log_dir
        self._client     = None

        self.action_space = spaces.Box(
            low=np.array([-DELTA_CLIP, -DELTA_CLIP, -DELTA_CLIP, -1.0], dtype=np.float32),
            high=np.array([DELTA_CLIP,  DELTA_CLIP,  DELTA_CLIP,  1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

        self._step_count    = 0
        self._episode_count = 0
        self._total_reward  = 0.0
        self._gripper_state = 1.0
        self._csv_writer    = None
        self._csv_file      = None

        self._contact_given    = False
        self._grasp_given      = False
        self._lift_given       = False
        self._close_near_given = False

        self._granger = NeuralGrangerCausalReward()

    def _connect(self):
        if self._client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)

    def _open_csv(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "episode_diagnostics.csv")
        new  = not os.path.exists(path)
        self._csv_file   = open(path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if new:
            self._csv_writer.writerow([
                "episode", "step",
                "ee_cube_dist", "cube_tray_dist", "cube_z",
                "any_contact", "grasp", "gripper_state",
                "reward", "total_reward",
            ])

    def _log_step(self, step, ee_cube_dist, cube_tray_dist,
                  cube_z, any_contact, grasp, reward, total_reward):
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            self._episode_count, step,
            f"{ee_cube_dist:.4f}", f"{cube_tray_dist:.4f}", f"{cube_z:.4f}",
            int(any_contact), int(grasp), f"{self._gripper_state:.3f}",
            f"{reward:.4f}", f"{total_reward:.4f}",
        ])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        if self._csv_writer is None:
            self._open_csv()

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        self._robot = p.loadURDF(
            ROBOT_URDF, basePosition=[0, 0, 0],
            useFixedBase=True, physicsClientId=self._client,
        )
        for i, angle in zip(ARM_JOINTS, REST_POSE):
            p.resetJointState(self._robot, i, angle, physicsClientId=self._client)
        for fj in FINGER_JOINTS:
            p.resetJointState(self._robot, fj, FINGER_OPEN, physicsClientId=self._client)
        self._gripper_state = 1.0

        x = random.uniform(0.35, 0.65)
        y = random.uniform(-0.25, 0.25)
        self._object_id = p.loadURDF(
            "cube_small.urdf", basePosition=[x, y, 0.02],
            physicsClientId=self._client,
        )
        self._container = p.loadURDF(
            "tray/traybox.urdf", basePosition=[0.5, 0.4, 0.0],
            physicsClientId=self._client,
        )
        tray_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)
        self._tray_z = tray_pos[2]

        self._step_count       = 0
        self._episode_count   += 1
        self._total_reward     = 0.0
        self._contact_given    = False
        self._grasp_given      = False
        self._lift_given       = False
        self._close_near_given = False
        self._granger.reset()

        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)
        return self._get_obs(), {}

    def step(self, action):
        delta_ee    = action[:3]
        gripper_cmd = float(action[3])

        ee_pos    = np.array(self._get_ee_pos())
        target_ee = np.clip(ee_pos + delta_ee, WORKSPACE["low"], WORKSPACE["high"])
        joint_poses = p.calculateInverseKinematics(
            self._robot, EE_LINK, target_ee.tolist(),
            restPoses=REST_POSE + [0.04, 0.04],
            physicsClientId=self._client,
        )
        for j in ARM_JOINTS:
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=200,
                physicsClientId=self._client,
            )

        self._gripper_state = float(np.clip((gripper_cmd + 1.0) / 2.0, 0.0, 1.0))
        finger_pos = FINGER_CLOSED + self._gripper_state * (FINGER_OPEN - FINGER_CLOSED)
        for fj in FINGER_JOINTS:
            p.setJointMotorControl2(
                self._robot, fj,
                controlMode=p.POSITION_CONTROL,
                targetPosition=finger_pos,
                force=20,
                physicsClientId=self._client,
            )

        for _ in range(N_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._client)
        self._step_count += 1

        obs         = self._get_obs()
        ee_pos      = obs[14:17]
        obj_pos     = obs[17:20]
        tray_pos    = obs[20:23]
        cube_z      = float(obs[23])
        any_contact = bool(obs[24])
        grasp       = bool(obs[25])

        ee_cube_dist = float(np.linalg.norm(ee_pos - obj_pos))
        granger_r    = self._granger.update_and_compute(
            self._gripper_state, obj_pos, ee_cube_dist)

        reward, terminated = self._compute_reward(
            ee_pos, obj_pos, tray_pos, cube_z, any_contact, grasp, granger_r)
        self._total_reward += reward
        truncated = self._step_count >= MAX_STEPS

        cube_tray_dist = float(np.linalg.norm(obj_pos - tray_pos))
        self._log_step(self._step_count, ee_cube_dist, cube_tray_dist,
                       cube_z, any_contact, grasp, reward, self._total_reward)
        if terminated or truncated:
            self._csv_file.flush()

        return obs, reward, terminated, truncated, {
            "ee_cube_dist":   ee_cube_dist,
            "cube_tray_dist": cube_tray_dist,
            "cube_z":         cube_z,
            "any_contact":    any_contact,
            "grasp":          grasp,
            "gripper_state":  self._gripper_state,
        }

    def _get_ee_pos(self):
        ls = p.getLinkState(self._robot, EE_LINK, physicsClientId=self._client)
        return list(ls[0])

    def _get_obs(self):
        js = [p.getJointState(self._robot, i, physicsClientId=self._client)
              for i in ARM_JOINTS]
        arm_angles = [s[0] for s in js]
        arm_vels   = [s[1] for s in js]
        ee_pos        = self._get_ee_pos()
        obj_pos, _    = p.getBasePositionAndOrientation(
            self._object_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)
        cube_z = float(obj_pos[2])

        any_contacts = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            physicsClientId=self._client)
        any_contact = float(len(any_contacts) > 0)

        c_f1 = p.getContactPoints(self._robot, self._object_id,
                                   linkIndexA=FINGER_LINKS[0],
                                   physicsClientId=self._client)
        c_f2 = p.getContactPoints(self._robot, self._object_id,
                                   linkIndexA=FINGER_LINKS[1],
                                   physicsClientId=self._client)
        grasp = float(len(c_f1) > 0 and len(c_f2) > 0
                      and self._gripper_state < 0.5)

        obs = (arm_angles + arm_vels
               + ee_pos + list(obj_pos) + list(target_pos)
               + [cube_z, any_contact, grasp, self._gripper_state])
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, ee_pos, obj_pos, tray_pos,
                        cube_z, any_contact, grasp, granger_r):
        dist_ee_obj   = float(np.linalg.norm(ee_pos  - obj_pos))
        dist_obj_tray = float(np.linalg.norm(obj_pos - tray_pos))
        reward = 0.0

        # Dense base
        reward -= 1.0 * dist_ee_obj
        reward -= 0.2 * dist_obj_tray
        reward -= 0.01

        # Dense gripper shaping
        if dist_ee_obj < GRIPPER_PROXIMITY:
            proximity = 1.0 - (dist_ee_obj / GRIPPER_PROXIMITY)
            closure   = 1.0 - self._gripper_state
            reward   += 0.3 * proximity * closure

        # True Granger-causal reward
        reward += granger_r

        # One-time bonuses
        if any_contact and not self._contact_given:
            reward += 5.0
            self._contact_given = True
        if dist_ee_obj < GRIPPER_PROXIMITY and self._gripper_state < 0.4 \
                and not self._close_near_given:
            reward += 8.0
            self._close_near_given = True
        if grasp and not self._grasp_given:
            reward += 15.0
            self._grasp_given = True
        if cube_z > LIFT_THRESHOLD and not self._lift_given:
            reward += 20.0
            self._lift_given = True

        # Conditional transport
        if cube_z > LIFT_THRESHOLD:
            reward += 0.2 * dist_obj_tray
            reward -= 0.8 * dist_obj_tray

        # Terminal
        terminated = False
        if dist_obj_tray < PLACE_THRESHOLD and cube_z > self._tray_z + 0.03:
            reward    += 100.0
            terminated = True
        if cube_z < -0.10:
            reward    -= 10.0
            terminated = True

        return reward, terminated

    def render(self):
        pass

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None
