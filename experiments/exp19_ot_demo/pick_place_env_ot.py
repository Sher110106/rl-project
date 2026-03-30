"""
Experiment 19: Self-Bootstrapped Optimal Transport Demo Reward
--------------------------------------------------------------
Implements Rank 5 from RESEARCH_AND_PLAN.md.

Core idea: use our own successful episodes (from Exp5/14) as demonstrations.
At each step, reward the agent for having its current state close to any
state seen in a successful trajectory — self-improving imitation.

"Optimal Transport" connection:
  OT distance between agent's trajectory distribution and demo distribution
  = Wasserstein distance W(τ_agent, τ_demo) in state space.
  A step-level proxy: r_ot(s) ∝ -min_dist(s, demo_states)
  This lower-bounds the full trajectory OT reward.

Feature space (6D):
  (ee_cube_dist, cube_tray_dist, cube_z, any_contact, grasp, gripper_state)
  — fully computable from env observations at every step
  — normalised to [0, 1] range using stats from demo distribution

Nearest-neighbour lookup via sklearn BallTree (O(log N), N=10k demo states).

Reward:
  r_ot = max(0, OT_SIGMA - nn_dist) / OT_SIGMA * OT_SCALE
       = 0 when far from any demo state
       = OT_SCALE when exactly on a demo state

Combined with:
  - Dense distance shaping (reach + tray pull)
  - Dense gripper proximity shaping
  - Granger(scale=0.8) causal signal
  - One-time bonuses
"""

import csv
import os
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from sklearn.neighbors import BallTree

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

# Granger config (carried from exp14 best)
CAUSAL_WINDOW    = 8
CAUSAL_SCALE     = 0.8
CAUSAL_PROXIMITY = 0.12

# OT reward config
OT_SIGMA  = 0.15   # neighbourhood radius — states within this distance get reward
OT_SCALE  = 0.3    # max per-step OT bonus (comparable to gripper shaping)
# Feature normalisation ranges (from demo distribution)
OT_NORM = np.array([
    0.65,   # ee_cube_dist   max ~0.65
    0.70,   # cube_tray_dist max ~0.70
    0.12,   # cube_z         max ~0.12
    1.0,    # any_contact    binary
    1.0,    # grasp          binary
    1.0,    # gripper_state  [0,1]
], dtype=np.float32)


class GrangerCausalReward:
    def __init__(self, window=CAUSAL_WINDOW):
        self._gripper_hist = deque(maxlen=window)
        self._cube_hist    = deque(maxlen=window)

    def reset(self):
        self._gripper_hist.clear()
        self._cube_hist.clear()

    def update(self, gripper_state, cube_pos):
        self._gripper_hist.append(float(gripper_state))
        self._cube_hist.append(cube_pos.copy())

    def compute(self, ee_cube_dist):
        if ee_cube_dist > CAUSAL_PROXIMITY or len(self._gripper_hist) < 4:
            return 0.0
        g  = np.array(list(self._gripper_hist))
        c  = np.array(list(self._cube_hist))
        gd = np.diff(g)
        cd = np.linalg.norm(np.diff(c, axis=0), axis=1)
        n  = min(len(gd), len(cd), 4)
        gd, cd = gd[-n:], cd[-n:]
        if np.std(gd) < 1e-7 or np.std(cd) < 1e-7:
            return 0.0
        corr = np.corrcoef(gd, cd)[0, 1]
        return float(max(0.0, 0.0 if np.isnan(corr) else corr)) * CAUSAL_SCALE


class OTDemoReward:
    """
    Nearest-neighbour demo reward in normalised 6D feature space.
    Loaded once at init; BallTree for O(log N) per-step lookup.
    """

    def __init__(self, demo_path: str):
        demos_raw = np.load(demo_path).astype(np.float32)
        # Normalise features to [0,1]
        self._demos_norm = demos_raw / OT_NORM[np.newaxis, :]
        self._tree = BallTree(self._demos_norm, metric="euclidean")
        print(f"[OTDemoReward] Loaded {len(self._demos_norm)} demo states "
              f"from {demo_path}")

    def _features(self, ee_cube_dist, cube_tray_dist, cube_z,
                  any_contact, grasp, gripper_state):
        feat = np.array([ee_cube_dist, cube_tray_dist, cube_z,
                         float(any_contact), float(grasp), float(gripper_state)],
                        dtype=np.float32)
        return feat / OT_NORM

    def compute(self, ee_cube_dist, cube_tray_dist, cube_z,
                any_contact, grasp, gripper_state):
        """Return OT demo reward in [0, OT_SCALE]."""
        feat = self._features(ee_cube_dist, cube_tray_dist, cube_z,
                              any_contact, grasp, gripper_state)
        dist, _ = self._tree.query(feat.reshape(1, -1), k=1)
        nn_dist = float(dist[0, 0])
        bonus = max(0.0, OT_SIGMA - nn_dist) / OT_SIGMA * OT_SCALE
        return bonus


class PickPlaceOTEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, log_dir="logs/diagnostics",
                 demo_path="ot_demos.npy"):
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

        self._causal  = GrangerCausalReward()
        self._ot      = OTDemoReward(demo_path)

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
                "episode", "step", "ee_cube_dist", "cube_tray_dist", "cube_z",
                "any_contact", "grasp", "gripper_state", "reward", "total_reward",
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
        self._causal.reset()

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

        ee_cube_dist   = float(np.linalg.norm(ee_pos  - obj_pos))
        cube_tray_dist = float(np.linalg.norm(obj_pos - tray_pos))

        self._causal.update(self._gripper_state, obj_pos)

        reward, terminated = self._compute_reward(
            ee_pos, obj_pos, tray_pos, cube_z, any_contact, grasp,
            ee_cube_dist, cube_tray_dist)
        self._total_reward += reward
        truncated = self._step_count >= MAX_STEPS

        self._log_step(self._step_count, ee_cube_dist, cube_tray_dist,
                       cube_z, any_contact, grasp, reward, self._total_reward)
        if terminated or truncated:
            self._csv_file.flush()

        return obs, reward, terminated, truncated, {
            "ee_cube_dist": ee_cube_dist, "cube_tray_dist": cube_tray_dist,
            "cube_z": cube_z, "any_contact": any_contact,
            "grasp": grasp, "gripper_state": self._gripper_state,
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
                        cube_z, any_contact, grasp,
                        ee_cube_dist, cube_tray_dist):
        reward = 0.0

        # Dense base
        reward -= 1.0 * ee_cube_dist
        reward -= 0.2 * cube_tray_dist
        reward -= 0.01

        # Dense gripper shaping
        if ee_cube_dist < GRIPPER_PROXIMITY:
            proximity = 1.0 - (ee_cube_dist / GRIPPER_PROXIMITY)
            closure   = 1.0 - self._gripper_state
            reward   += 0.3 * proximity * closure

        # Granger causal signal
        reward += self._causal.compute(ee_cube_dist)

        # OT demo reward — pulls agent toward successful-episode state distribution
        reward += self._ot.compute(ee_cube_dist, cube_tray_dist, cube_z,
                                   any_contact, grasp, self._gripper_state)

        # One-time bonuses
        if any_contact and not self._contact_given:
            reward += 5.0
            self._contact_given = True
        if ee_cube_dist < GRIPPER_PROXIMITY and self._gripper_state < 0.4 \
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
            reward += 0.2 * cube_tray_dist
            reward -= 0.8 * cube_tray_dist

        # Terminal
        terminated = False
        if cube_tray_dist < PLACE_THRESHOLD and cube_z > self._tray_z + 0.03:
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
