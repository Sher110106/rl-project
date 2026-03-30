"""
Experiment 3: Curriculum Learning
-----------------------------------
Progressive difficulty — start easy, advance when competent.

Stage 1 — "Learn to grasp" (easy)
    Cube spawns CLOSE to tray (0.05-0.15 m away)
    EE starts directly ABOVE the cube (via IK)
    Agent only needs: lower → close gripper → tiny lift → done
    Probability of accidental success is high → learns fast

Stage 2 — "Learn to reach + grasp" (medium)
    Cube spawns randomly (normal range)
    EE starts above the cube (via IK)
    Agent must: descend → grasp → lift → transport to tray

Stage 3 — "Full task" (hard)
    Cube random, EE at rest pose
    Agent must: reach → grasp → lift → transport → place

Auto-advances when success rate > 15% over last 100 episodes.

State (27-dim), Action (4-dim), Reward: same as v4
"""

import csv
import os
import random
from collections import deque

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

PLACE_THRESHOLD = 0.05
LIFT_THRESHOLD  = 0.05
MAX_STEPS       = 500
DELTA_CLIP      = 0.05
N_SUBSTEPS      = 10
ADVANCE_THRESHOLD = 0.15   # success rate to advance stage
WORKSPACE = {
    "low":  np.array([0.25, -0.55, 0.01]),
    "high": np.array([0.80,  0.55, 0.70]),
}


class PickPlaceCurriculumEnv(gym.Env):
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

        # Curriculum state
        self._stage          = 1
        self._success_buffer = deque(maxlen=100)

        # One-time flags
        self._contact_given = False
        self._grasp_given   = False
        self._lift_given    = False

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
                physicsClientId=self._client)

    def _open_csv(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "episode_diagnostics.csv")
        new  = not os.path.exists(path)
        self._csv_file   = open(path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if new:
            self._csv_writer.writerow([
                "episode", "step", "stage",
                "ee_cube_dist", "cube_tray_dist",
                "cube_z", "any_contact", "grasp", "gripper_state",
                "reward", "total_reward",
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
            useFixedBase=True, physicsClientId=self._client)

        # ── Stage-dependent initial conditions ─────────────────────
        tray_x, tray_y = 0.5, 0.4
        self._container = p.loadURDF(
            "tray/traybox.urdf", basePosition=[tray_x, tray_y, 0.0],
            physicsClientId=self._client)
        tray_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)
        self._tray_z = tray_pos[2]

        if self._stage == 1:
            # Cube CLOSE to tray
            cx = tray_x + random.uniform(-0.10, 0.10)
            cy = tray_y + random.uniform(-0.10, 0.10)
        elif self._stage == 2:
            # Cube random (normal range)
            cx = random.uniform(0.35, 0.65)
            cy = random.uniform(-0.25, 0.25)
        else:
            # Full random
            cx = random.uniform(0.35, 0.65)
            cy = random.uniform(-0.25, 0.25)

        self._object_id = p.loadURDF(
            "cube_small.urdf", basePosition=[cx, cy, 0.02],
            physicsClientId=self._client)

        if self._stage <= 2:
            # EE starts ABOVE the cube via IK
            target_above = [cx, cy, 0.15]
            ik_joints = p.calculateInverseKinematics(
                self._robot, EE_LINK, target_above,
                restPoses=REST_POSE + [0.04, 0.04],
                physicsClientId=self._client)
            for j in ARM_JOINTS:
                p.resetJointState(self._robot, j, ik_joints[j],
                                  physicsClientId=self._client)
        else:
            # Rest pose
            for i, angle in zip(ARM_JOINTS, REST_POSE):
                p.resetJointState(self._robot, i, angle,
                                  physicsClientId=self._client)

        # Gripper open
        for fj in FINGER_JOINTS:
            p.resetJointState(self._robot, fj, FINGER_OPEN,
                              physicsClientId=self._client)
        self._gripper_state = 1.0

        self._step_count    = 0
        self._episode_count += 1
        self._total_reward  = 0.0
        self._contact_given = False
        self._grasp_given   = False
        self._lift_given    = False
        self._ep_success    = False

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
            physicsClientId=self._client)
        for j in ARM_JOINTS:
            p.setJointMotorControl2(
                self._robot, j, p.POSITION_CONTROL,
                targetPosition=joint_poses[j], force=200,
                physicsClientId=self._client)

        self._gripper_state = float(np.clip((gripper_cmd + 1.0) / 2.0, 0.0, 1.0))
        finger_pos = FINGER_CLOSED + self._gripper_state * (FINGER_OPEN - FINGER_CLOSED)
        for fj in FINGER_JOINTS:
            p.setJointMotorControl2(
                self._robot, fj, p.POSITION_CONTROL,
                targetPosition=finger_pos, force=20,
                physicsClientId=self._client)

        for _ in range(N_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._client)
        self._step_count += 1

        obs      = self._get_obs()
        ee_pos   = obs[14:17]
        obj_pos  = obs[17:20]
        tray_pos = obs[20:23]
        cube_z   = float(obs[23])
        any_contact = bool(obs[24])
        grasp       = bool(obs[25])

        reward, terminated = self._compute_reward(
            ee_pos, obj_pos, tray_pos, cube_z, any_contact, grasp)
        self._total_reward += reward
        truncated = self._step_count >= MAX_STEPS

        ee_cube_dist   = float(np.linalg.norm(ee_pos  - obj_pos))
        cube_tray_dist = float(np.linalg.norm(obj_pos - tray_pos))

        if self._csv_writer:
            self._csv_writer.writerow([
                self._episode_count, self._step_count, self._stage,
                f"{ee_cube_dist:.4f}", f"{cube_tray_dist:.4f}",
                f"{cube_z:.4f}", int(any_contact), int(grasp),
                f"{self._gripper_state:.3f}",
                f"{reward:.4f}", f"{self._total_reward:.4f}"])
            if terminated or truncated:
                self._csv_file.flush()

        # ── Curriculum advancement ─────────────────────────────────
        if terminated or truncated:
            self._success_buffer.append(self._ep_success)
            if (len(self._success_buffer) == self._success_buffer.maxlen
                    and self._stage < 3):
                rate = sum(self._success_buffer) / len(self._success_buffer)
                if rate >= ADVANCE_THRESHOLD:
                    self._stage += 1
                    self._success_buffer.clear()
                    print(f"\n*** CURRICULUM → Stage {self._stage} "
                          f"(success rate was {rate:.0%}) ***\n")

        return obs, reward, terminated, truncated, {
            "ee_cube_dist":   ee_cube_dist,
            "cube_tray_dist": cube_tray_dist,
            "cube_z":         cube_z,
            "any_contact":    any_contact,
            "grasp":          grasp,
            "gripper_state":  self._gripper_state,
            "stage":          self._stage,
            "is_success":     self._ep_success,
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

        c1 = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            linkIndexA=FINGER_LINKS[0], physicsClientId=self._client)
        c2 = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            linkIndexA=FINGER_LINKS[1], physicsClientId=self._client)
        grasp = float(len(c1) > 0 and len(c2) > 0 and self._gripper_state < 0.5)

        obs = (arm_angles + arm_vels + ee_pos
               + list(obj_pos) + list(target_pos)
               + [cube_z, any_contact, grasp, self._gripper_state])
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, ee_pos, obj_pos, tray_pos, cube_z, any_contact, grasp):
        dist_ee_obj   = np.linalg.norm(ee_pos  - obj_pos)
        dist_obj_tray = np.linalg.norm(obj_pos - tray_pos)

        reward = 0.0

        reward -= 1.0 * dist_ee_obj
        reward -= 0.2 * dist_obj_tray
        reward -= 0.01

        if any_contact and not self._contact_given:
            reward += 5.0
            self._contact_given = True

        if grasp and not self._grasp_given:
            reward += 15.0
            self._grasp_given = True

        if cube_z > LIFT_THRESHOLD and not self._lift_given:
            reward += 20.0
            self._lift_given = True

        if cube_z > LIFT_THRESHOLD:
            reward += 0.2 * dist_obj_tray
            reward -= 1.0 * dist_obj_tray

        terminated = False
        if dist_obj_tray < PLACE_THRESHOLD and cube_z > self._tray_z + 0.03:
            reward    += 100.0
            terminated = True
            self._ep_success = True

        if cube_z < -0.10:
            reward    -= 10.0
            terminated = True

        return reward, terminated

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
