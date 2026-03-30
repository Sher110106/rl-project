"""
Experiment 1: HER + SAC — Goal-Conditioned Pick-and-Place
---------------------------------------------------------
Uses Hindsight Experience Replay (HER) with sparse reward.

HER replaces the desired goal with the achieved goal in failed
trajectories, converting failures into synthetic successes. This
is the standard approach for sparse-reward manipulation (OpenAI, 2018).

Observation (Dict):
    "observation"   (21): arm_angles(7) + arm_vels(7) + ee_pos(3) +
                          gripper_state(1) + any_contact(1) + grasp(1) + cube_z(1)
    "achieved_goal" (3):  current cube position [x, y, z]
    "desired_goal"  (3):  target position [tray_x, tray_y, tray_z + 0.05]

Action (4): [dx, dy, dz, gripper]

Reward: SPARSE
    0  if dist(achieved, desired) < 0.05  (success)
   -1  otherwise
"""

import csv
import os
import random

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

ROBOT_URDF     = "franka_panda/panda.urdf"
ARM_JOINTS     = list(range(7))
FINGER_JOINTS  = [9, 10]
FINGER_LINKS   = [9, 10]
EE_LINK        = 11
FINGER_OPEN    = 0.04
FINGER_CLOSED  = 0.00
REST_POSE      = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

SUCCESS_THRESHOLD = 0.05
LIFT_THRESHOLD    = 0.05
MAX_STEPS         = 500
DELTA_CLIP        = 0.05
N_SUBSTEPS        = 10
WORKSPACE = {
    "low":  np.array([0.25, -0.55, 0.01]),
    "high": np.array([0.80,  0.55, 0.70]),
}


class PickPlaceHEREnv(gym.Env):
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

        # HER requires Dict observation space
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,),  dtype=np.float32),
            "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(3,),  dtype=np.float32),
        })

        self._step_count    = 0
        self._episode_count = 0
        self._gripper_state = 1.0
        self._csv_writer    = None
        self._csv_file      = None

    def _connect(self):
        if self._client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0.4, 0.0, 0.2],
                physicsClientId=self._client,
            )

    def _open_csv(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "episode_diagnostics.csv")
        new  = not os.path.exists(path)
        self._csv_file   = open(path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if new:
            self._csv_writer.writerow([
                "episode", "step",
                "ee_cube_dist", "cube_tray_dist",
                "cube_z", "any_contact", "grasp", "gripper_state",
                "reward", "is_success",
            ])

    # ── HER required: compute_reward for batched goals ─────────────────
    def compute_reward(self, achieved_goal, desired_goal, _info):
        """Sparse reward: 0 if close enough, -1 otherwise. Handles batches."""
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > SUCCESS_THRESHOLD).astype(np.float32)

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
            p.resetJointState(self._robot, i, angle,
                              physicsClientId=self._client)
        for fj in FINGER_JOINTS:
            p.resetJointState(self._robot, fj, FINGER_OPEN,
                              physicsClientId=self._client)
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
        # Desired goal: above the tray centre
        self._desired_goal = np.array(
            [tray_pos[0], tray_pos[1], tray_pos[2] + 0.05], dtype=np.float32
        )

        self._step_count    = 0
        self._episode_count += 1

        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)

        return self._get_obs_dict(), {}

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
                self._robot, j, p.POSITION_CONTROL,
                targetPosition=joint_poses[j], force=200,
                physicsClientId=self._client,
            )

        self._gripper_state = float(np.clip((gripper_cmd + 1.0) / 2.0, 0.0, 1.0))
        finger_pos = FINGER_CLOSED + self._gripper_state * (FINGER_OPEN - FINGER_CLOSED)
        for fj in FINGER_JOINTS:
            p.setJointMotorControl2(
                self._robot, fj, p.POSITION_CONTROL,
                targetPosition=finger_pos, force=20,
                physicsClientId=self._client,
            )

        for _ in range(N_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._client)
        self._step_count += 1

        obs_dict = self._get_obs_dict()
        achieved = obs_dict["achieved_goal"]
        desired  = obs_dict["desired_goal"]

        reward = float(self.compute_reward(achieved, desired, {}))
        is_success = bool(reward == 0.0)

        terminated = is_success
        # Also terminate if cube falls off table
        cube_z = float(achieved[2])
        if cube_z < -0.10:
            terminated = True

        truncated = self._step_count >= MAX_STEPS

        ee_pos  = obs_dict["observation"][14:17]
        obj_pos = achieved
        ee_cube_dist   = float(np.linalg.norm(ee_pos - obj_pos))
        cube_tray_dist = float(np.linalg.norm(obj_pos - desired))
        any_contact    = bool(obs_dict["observation"][18])
        grasp          = bool(obs_dict["observation"][19])

        if self._csv_writer:
            self._csv_writer.writerow([
                self._episode_count, self._step_count,
                f"{ee_cube_dist:.4f}", f"{cube_tray_dist:.4f}",
                f"{cube_z:.4f}", int(any_contact), int(grasp),
                f"{self._gripper_state:.3f}",
                f"{reward:.1f}", int(is_success),
            ])
            if terminated or truncated:
                self._csv_file.flush()

        info = {
            "is_success":     is_success,
            "ee_cube_dist":   ee_cube_dist,
            "cube_tray_dist": cube_tray_dist,
            "cube_z":         cube_z,
            "any_contact":    any_contact,
            "grasp":          grasp,
        }

        return obs_dict, reward, terminated, truncated, info

    def _get_ee_pos(self):
        ls = p.getLinkState(self._robot, EE_LINK, physicsClientId=self._client)
        return list(ls[0])

    def _get_obs_dict(self):
        js = [p.getJointState(self._robot, i, physicsClientId=self._client)
              for i in ARM_JOINTS]
        arm_angles = [s[0] for s in js]
        arm_vels   = [s[1] for s in js]

        ee_pos = self._get_ee_pos()

        obj_pos, _ = p.getBasePositionAndOrientation(
            self._object_id, physicsClientId=self._client)
        cube_z = float(obj_pos[2])

        any_contacts = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            physicsClientId=self._client)
        any_contact = float(len(any_contacts) > 0)

        c1 = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            linkIndexA=FINGER_LINKS[0], physicsClientId=self._client)
        c2 = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            linkIndexA=FINGER_LINKS[1], physicsClientId=self._client)
        grasp = float(len(c1) > 0 and len(c2) > 0 and self._gripper_state < 0.5)

        observation = np.array(
            arm_angles + arm_vels + ee_pos
            + [self._gripper_state, any_contact, grasp, cube_z],
            dtype=np.float32,
        )
        achieved_goal = np.array(obj_pos, dtype=np.float32)

        return {
            "observation":   observation,
            "achieved_goal": achieved_goal,
            "desired_goal":  self._desired_goal.copy(),
        }

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
