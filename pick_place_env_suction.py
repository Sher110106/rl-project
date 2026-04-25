"""
pick_place_env_suction.py
-------------------------
PyBullet pick-and-place with Kuka IIWA + suction gripper.

State (24-dim):
    joint_angles(7) + joint_velocities(7) + ee_pos(3) +
    object_pos(3) + tray_pos(3) + suction_on(1)

Action (4-dim):
    [dx, dy, dz, suction_action]
    dx,dy,dz ∈ [-0.05, 0.05] m per action (IK-resolved)
    suction_action > 0  → attempt to turn suction ON (if close to cube)
    suction_action <= 0 → turn suction OFF

Reward:
    - dist(EE → cube) - dist(cube → tray)
    + 10  first successful suction grasp (one-time per episode)
    + 50  successful placement (cube within 0.05 m of tray, one-time)

Suction mechanism:
    PyBullet JOINT_FIXED constraint between EE link and cube.
    Activation requires: dist(EE, cube) < 0.08 m.
"""

import csv
import os
import random

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

# ── Constants ──────────────────────────────────────────────────────────────
ROBOT_URDF = "kuka_iiwa/model.urdf"
ARM_JOINTS = list(range(7))
MAX_STEPS = 500
DELTA_CLIP = 0.05
N_SUBSTEPS = 10
SUCCESS_THRESHOLD = 0.05
SUCTION_THRESHOLD = 0.08
SUCTION_BONUS = 10.0
SUCCESS_BONUS = 50.0
WORKSPACE = {
    "low": np.array([0.25, -0.55, 0.01]),
    "high": np.array([0.80, 0.55, 0.70]),
}


class PickPlaceSuctionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, log_dir="logs/diagnostics"):
        super().__init__()
        self.render_mode = render_mode
        self.log_dir = log_dir
        self._client = None

        # 4D action: [dx, dy, dz, suction]
        self.action_space = spaces.Box(
            low=np.array([-DELTA_CLIP, -DELTA_CLIP, -DELTA_CLIP, -1.0], dtype=np.float32),
            high=np.array([DELTA_CLIP, DELTA_CLIP, DELTA_CLIP, 1.0], dtype=np.float32),
        )

        # 24D observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

        self._step_count = 0
        self._episode_count = 0
        self._total_reward = 0.0
        self._suction_on = 0.0
        self._constraint_id = None

        # One-time bonus flags
        self._suction_bonus_given = False
        self._success_bonus_given = False

        self._csv_writer = None
        self._csv_file = None

    # ── PyBullet connection ────────────────────────────────────────────────
    def _connect(self):
        if self._client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client
        )
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.4, 0.0, 0.2],
                physicsClientId=self._client,
            )

    # ── CSV logging ────────────────────────────────────────────────────────
    def _open_csv(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "episode_diagnostics.csv")
        new = not os.path.exists(path)
        self._csv_file = open(path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if new:
            self._csv_writer.writerow(
                [
                    "episode",
                    "step",
                    "ee_cube_dist",
                    "cube_tray_dist",
                    "cube_z",
                    "suction_on",
                    "reward",
                    "total_reward",
                ]
            )

    def _log_step(self, step, ee_cube_dist, cube_tray_dist, cube_z, reward, total_reward):
        if self._csv_writer is None:
            return
        self._csv_writer.writerow(
            [
                self._episode_count,
                step,
                f"{ee_cube_dist:.4f}",
                f"{cube_tray_dist:.4f}",
                f"{cube_z:.4f}",
                int(self._suction_on),
                f"{reward:.4f}",
                f"{total_reward:.4f}",
            ]
        )

    # ── Reset ──────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()

        if self._csv_writer is None:
            self._open_csv()

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)

        p.loadURDF("plane.urdf", physicsClientId=self._client)
        self._robot = p.loadURDF(
            ROBOT_URDF,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client,
        )

        # Kuka has 7 joints; reset to neutral pose
        self._num_joints = p.getNumJoints(self._robot, physicsClientId=self._client)
        self._ee_link = self._num_joints - 1  # last link = end-effector
        for j in ARM_JOINTS:
            p.resetJointState(self._robot, j, 0.0, physicsClientId=self._client)

        # Randomise cube position
        x = random.uniform(0.35, 0.65)
        y = random.uniform(-0.25, 0.25)
        self._object_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[x, y, 0.02],
            physicsClientId=self._client,
        )
        self._container = p.loadURDF(
            "tray/traybox.urdf",
            basePosition=[0.5, 0.4, 0.0],
            physicsClientId=self._client,
        )

        # Reset episode tracking
        self._step_count = 0
        self._episode_count += 1
        self._total_reward = 0.0
        self._suction_on = 0.0
        self._constraint_id = None
        self._suction_bonus_given = False
        self._success_bonus_given = False

        # Warm-up physics
        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)

        return self._get_obs(), {}

    # ── Step ───────────────────────────────────────────────────────────────
    def step(self, action):
        delta_ee = action[:3]
        suction_cmd = float(action[3])

        # ── EE control via IK ──
        ee_pos = np.array(self._get_ee_pos())
        target_ee = np.clip(ee_pos + delta_ee, WORKSPACE["low"], WORKSPACE["high"])
        joint_poses = p.calculateInverseKinematics(
            self._robot, self._ee_link, target_ee.tolist(),
            restPoses=[0.0] * self._num_joints,
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

        # ── Suction logic ────────────────────────────────────────────────
        obj_pos, _ = p.getBasePositionAndOrientation(
            self._object_id, physicsClientId=self._client
        )
        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))

        if suction_cmd > 0.0 and self._suction_on == 0.0:
            # Attempt to activate suction
            if dist_ee_obj < SUCTION_THRESHOLD:
                self._constraint_id = p.createConstraint(
                    parentBodyUniqueId=self._robot,
                    parentLinkIndex=self._ee_link,
                    childBodyUniqueId=self._object_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                    physicsClientId=self._client,
                )
                self._suction_on = 1.0

        elif suction_cmd <= 0.0 and self._suction_on == 1.0:
            # Turn suction off
            if self._constraint_id is not None:
                p.removeConstraint(
                    self._constraint_id, physicsClientId=self._client
                )
                self._constraint_id = None
            self._suction_on = 0.0

        # Step physics
        for _ in range(N_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._client)
        self._step_count += 1

        # ── Observation ──────────────────────────────────────────────────
        obs = self._get_obs()
        ee_pos = obs[14:17]
        obj_pos = obs[17:20]
        tray_pos = obs[20:23]
        cube_z = float(obj_pos[2])

        # Recompute dist after physics step
        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))

        # ── Reward ───────────────────────────────────────────────────────
        reward, terminated = self._compute_reward(
            ee_pos, obj_pos, tray_pos, dist_ee_obj
        )
        self._total_reward += reward
        truncated = self._step_count >= MAX_STEPS

        ee_cube_dist = float(np.linalg.norm(ee_pos - obj_pos))
        cube_tray_dist = float(np.linalg.norm(obj_pos - tray_pos))
        self._log_step(
            self._step_count, ee_cube_dist, cube_tray_dist, cube_z, reward, self._total_reward
        )

        if terminated or truncated:
            self._csv_file.flush()

        info = {
            "ee_cube_dist": ee_cube_dist,
            "cube_tray_dist": cube_tray_dist,
            "cube_z": cube_z,
            "suction_on": bool(self._suction_on),
        }
        return obs, reward, terminated, truncated, info

    # ── Observation ────────────────────────────────────────────────────────
    def _get_ee_pos(self):
        ls = p.getLinkState(
            self._robot, self._ee_link, physicsClientId=self._client
        )
        return list(ls[0])

    def _get_obs(self):
        # Joint states
        joint_states = [
            p.getJointState(self._robot, j, physicsClientId=self._client)
            for j in ARM_JOINTS
        ]
        joint_angles = [s[0] for s in joint_states]
        joint_vels = [s[1] for s in joint_states]

        # Positions
        ee_pos = self._get_ee_pos()
        obj_pos, _ = p.getBasePositionAndOrientation(
            self._object_id, physicsClientId=self._client
        )
        tray_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client
        )

        obs = (
            joint_angles
            + joint_vels
            + ee_pos
            + list(obj_pos)
            + list(tray_pos)
            + [self._suction_on]
        )
        return np.array(obs, dtype=np.float32)

    # ── Reward ─────────────────────────────────────────────────────────────
    def _compute_reward(self, ee_pos, obj_pos, tray_pos, dist_ee_obj):
        dist_obj_tray = float(np.linalg.norm(obj_pos - tray_pos))

        reward = 0.0
        # Dense shaping
        reward -= dist_ee_obj
        reward -= dist_obj_tray

        # One-time suction grasp bonus
        if self._suction_on == 1.0 and not self._suction_bonus_given:
            reward += SUCTION_BONUS
            self._suction_bonus_given = True

        # One-time success bonus
        terminated = False
        if dist_obj_tray < SUCCESS_THRESHOLD and not self._success_bonus_given:
            reward += SUCCESS_BONUS
            self._success_bonus_given = True
            terminated = True

        return reward, terminated

    # ── Render / Close ─────────────────────────────────────────────────────
    def render(self):
        pass

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None
