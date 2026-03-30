"""
Experiment 6: Curriculum Learning — Stability Fix
--------------------------------------------------
Addresses the main failure mode of Exp3 (11% success, then regression):

  1. ADVANCE_THRESHOLD: 0.15 → 0.30  (must be 30% success to advance)
  2. ADVANCE_WINDOW:    100  → 300    (measured over 300 episodes, not 100)
  3. REGRESS_THRESHOLD: new — if success < 5% for 200 eps, drop back one stage
  4. Min-stage-time:    new — must stay in a stage ≥ 100 episodes before advancing

Plus the dense gripper reward from Exp2 carried over (it was the best reach→grasp signal).

This directly addresses: "Curriculum too eager to advance → catastrophic forgetting"
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

# Curriculum parameters (fixed vs Exp3)
ADVANCE_THRESHOLD = 0.30   # was 0.15 — need 30% to advance
ADVANCE_WINDOW    = 300    # was 100 — measured over 300 episodes
REGRESS_THRESHOLD = 0.05   # new — regress if below 5%
REGRESS_WINDOW    = 200    # new — over 200 episodes
MIN_STAGE_EPISODES = 150   # new — stay in stage at least 150 episodes


class PickPlaceCurriculumFixEnv(gym.Env):
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
        self._stage              = 1
        self._success_buf_adv    = deque(maxlen=ADVANCE_WINDOW)
        self._success_buf_reg    = deque(maxlen=REGRESS_WINDOW)
        self._stage_episode_count = 0  # episodes spent in current stage

        # One-time flags
        self._contact_given    = False
        self._grasp_given      = False
        self._lift_given       = False
        self._close_near_given = False

    # ── connection ──────────────────────────────────────────────────────────
    def _connect(self):
        if self._client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)

    # ── CSV diagnostics ──────────────────────────────────────────────────────
    def _open_csv(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "episode_diagnostics.csv")
        new  = not os.path.exists(path)
        self._csv_file   = open(path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if new:
            self._csv_writer.writerow([
                "episode", "step", "stage",
                "ee_cube_dist", "cube_tray_dist", "cube_z",
                "any_contact", "grasp", "gripper_state",
                "reward", "total_reward",
            ])

    def _log_step(self, step, stage, ee_cube_dist, cube_tray_dist,
                  cube_z, any_contact, grasp, reward, total_reward):
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            self._episode_count, step, stage,
            f"{ee_cube_dist:.4f}", f"{cube_tray_dist:.4f}", f"{cube_z:.4f}",
            int(any_contact), int(grasp), f"{self._gripper_state:.3f}",
            f"{reward:.4f}", f"{total_reward:.4f}",
        ])

    # ── curriculum logic ─────────────────────────────────────────────────────
    def _update_curriculum(self, success: bool):
        self._success_buf_adv.append(float(success))
        self._success_buf_reg.append(float(success))
        self._stage_episode_count += 1

        if self._stage < 3:
            if (len(self._success_buf_adv) >= ADVANCE_WINDOW
                    and self._stage_episode_count >= MIN_STAGE_EPISODES
                    and np.mean(self._success_buf_adv) >= ADVANCE_THRESHOLD):
                self._stage += 1
                self._success_buf_adv.clear()
                self._success_buf_reg.clear()
                self._stage_episode_count = 0
                print(f"  [curriculum] Advancing to Stage {self._stage} "
                      f"(ep {self._episode_count})")

        elif self._stage > 1:
            if (len(self._success_buf_reg) >= REGRESS_WINDOW
                    and self._stage_episode_count >= MIN_STAGE_EPISODES
                    and np.mean(self._success_buf_reg) < REGRESS_THRESHOLD):
                self._stage -= 1
                self._success_buf_adv.clear()
                self._success_buf_reg.clear()
                self._stage_episode_count = 0
                print(f"  [curriculum] Regressing to Stage {self._stage} "
                      f"(ep {self._episode_count})")

    # ── stage-specific reset ─────────────────────────────────────────────────
    def _stage_reset(self):
        """Place cube and optionally EE according to current stage."""
        tray_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)

        if self._stage == 1:
            # Stage 1: cube near tray, EE starts just above cube
            offset_x = random.uniform(-0.12, 0.12)
            offset_y = random.uniform(-0.12, 0.12)
            cx = np.clip(tray_pos[0] + offset_x, 0.35, 0.65)
            cy = np.clip(tray_pos[1] + offset_y, -0.25, 0.25)
            p.resetBasePositionAndOrientation(
                self._object_id, [cx, cy, 0.02], [0, 0, 0, 1],
                physicsClientId=self._client)
            # IK to place EE above cube
            above_cube = [cx, cy, 0.15]
            joints = p.calculateInverseKinematics(
                self._robot, EE_LINK, above_cube,
                restPoses=REST_POSE + [0.04, 0.04],
                physicsClientId=self._client)
            for j in ARM_JOINTS:
                p.resetJointState(self._robot, j, joints[j],
                                  physicsClientId=self._client)

        elif self._stage == 2:
            # Stage 2: cube random, EE starts above cube
            cx = random.uniform(0.35, 0.65)
            cy = random.uniform(-0.25, 0.25)
            p.resetBasePositionAndOrientation(
                self._object_id, [cx, cy, 0.02], [0, 0, 0, 1],
                physicsClientId=self._client)
            above_cube = [cx, cy, 0.18]
            joints = p.calculateInverseKinematics(
                self._robot, EE_LINK, above_cube,
                restPoses=REST_POSE + [0.04, 0.04],
                physicsClientId=self._client)
            for j in ARM_JOINTS:
                p.resetJointState(self._robot, j, joints[j],
                                  physicsClientId=self._client)

        else:
            # Stage 3: fully random — arm at rest pose, cube anywhere
            cx = random.uniform(0.35, 0.65)
            cy = random.uniform(-0.25, 0.25)
            p.resetBasePositionAndOrientation(
                self._object_id, [cx, cy, 0.02], [0, 0, 0, 1],
                physicsClientId=self._client)
            for i, angle in zip(ARM_JOINTS, REST_POSE):
                p.resetJointState(self._robot, i, angle,
                                  physicsClientId=self._client)

    # ── reset ────────────────────────────────────────────────────────────────
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

        # Load objects (positions set in _stage_reset)
        self._object_id = p.loadURDF(
            "cube_small.urdf", basePosition=[0.5, 0.0, 0.02],
            physicsClientId=self._client,
        )
        self._container = p.loadURDF(
            "tray/traybox.urdf", basePosition=[0.5, 0.4, 0.0],
            physicsClientId=self._client,
        )
        tray_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)
        self._tray_z = tray_pos[2]

        self._stage_reset()

        self._step_count       = 0
        self._episode_count   += 1
        self._total_reward     = 0.0
        self._contact_given    = False
        self._grasp_given      = False
        self._lift_given       = False
        self._close_near_given = False

        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)
        return self._get_obs(), {}

    # ── step ─────────────────────────────────────────────────────────────────
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

        reward, terminated = self._compute_reward(
            ee_pos, obj_pos, tray_pos, cube_z, any_contact, grasp
        )
        self._total_reward += reward
        truncated = self._step_count >= MAX_STEPS

        # Curriculum update at episode end
        if terminated or truncated:
            success = terminated and float(np.linalg.norm(obj_pos - tray_pos)) < PLACE_THRESHOLD
            self._update_curriculum(success)
            self._csv_file.flush()

        ee_cube_dist   = float(np.linalg.norm(ee_pos  - obj_pos))
        cube_tray_dist = float(np.linalg.norm(obj_pos - tray_pos))
        self._log_step(self._step_count, self._stage, ee_cube_dist, cube_tray_dist,
                       cube_z, any_contact, grasp, reward, self._total_reward)

        return obs, reward, terminated, truncated, {
            "ee_cube_dist":   ee_cube_dist,
            "cube_tray_dist": cube_tray_dist,
            "cube_z":         cube_z,
            "any_contact":    any_contact,
            "grasp":          grasp,
            "gripper_state":  self._gripper_state,
            "stage":          self._stage,
        }

    # ── observation ──────────────────────────────────────────────────────────
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

    # ── reward ───────────────────────────────────────────────────────────────
    def _compute_reward(self, ee_pos, obj_pos, tray_pos,
                        cube_z, any_contact, grasp):
        dist_ee_obj   = float(np.linalg.norm(ee_pos  - obj_pos))
        dist_obj_tray = float(np.linalg.norm(obj_pos - tray_pos))
        reward = 0.0

        reward -= 1.0 * dist_ee_obj
        reward -= 0.2 * dist_obj_tray
        reward -= 0.01

        # Dense gripper from Exp2
        if dist_ee_obj < GRIPPER_PROXIMITY:
            proximity = 1.0 - (dist_ee_obj / GRIPPER_PROXIMITY)
            closure   = 1.0 - self._gripper_state
            reward   += 0.3 * proximity * closure

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

        if cube_z > LIFT_THRESHOLD:
            reward += 0.2 * dist_obj_tray
            reward -= 0.8 * dist_obj_tray

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
