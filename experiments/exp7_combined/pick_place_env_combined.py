"""
Experiment 7: Grand Unified — Dense + Barrier + Granger + Curriculum Fix
-------------------------------------------------------------------------
Combines all novel ideas from the research plan:

  1. Dense gripper shaping (Exp2) — reach→grasp coordination
  2. Barrier function phase-gates (Exp4) — penalise out-of-phase actions
  3. Granger-causal intrinsic reward (Exp5) — reward actual cube manipulation
  4. Fixed curriculum (Exp6) — stable stage advancement with regression guard

Reward per step:
  -1.0 * dist(EE→cube)                     base reach
  -0.2 * dist(cube→tray)                   base tray pull
  -0.01                                     time penalty
  +0.3 * proximity * closure  if<0.08m     dense gripper
  -0.5 * closure_when_far                  barrier B1
  -2.0 * premature_lift                    barrier B2
  +r_granger                               causal influence

One-time: +5 contact, +8 close-near, +15 grasp, +20 lift
Conditional: -0.8 * dist(cube→tray) when lifted
Terminal: +100 success, -10 fall

Curriculum: 3 stages with 30% advance threshold, 300-ep window,
            5% regress threshold, 200-ep window, 150-ep min-stay
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

# Barrier
BARRIER_CLOSURE_FAR_COEF = 0.5
BARRIER_PREMATURE_COEF   = 2.0
BARRIER_FAR_THRESHOLD    = 0.10

# Granger
CAUSAL_WINDOW    = 8
CAUSAL_SCALE     = 0.4
CAUSAL_PROXIMITY = 0.12

# Curriculum
ADVANCE_THRESHOLD  = 0.30
ADVANCE_WINDOW     = 300
REGRESS_THRESHOLD  = 0.05
REGRESS_WINDOW     = 200
MIN_STAGE_EPISODES = 150


class GrangerCausalReward:
    def __init__(self, window=CAUSAL_WINDOW):
        self._gripper_hist = deque(maxlen=window)
        self._cube_hist    = deque(maxlen=window)

    def reset(self):
        self._gripper_hist.clear()
        self._cube_hist.clear()

    def update(self, gripper_state: float, cube_pos: np.ndarray):
        self._gripper_hist.append(gripper_state)
        self._cube_hist.append(cube_pos.copy())

    def compute(self, ee_cube_dist: float) -> float:
        if ee_cube_dist > CAUSAL_PROXIMITY:
            return 0.0
        if len(self._gripper_hist) < 4:
            return 0.0
        g  = np.array(list(self._gripper_hist))
        c  = np.array(list(self._cube_hist))
        gd = np.diff(g)
        cd = np.linalg.norm(np.diff(c, axis=0), axis=1)
        n  = min(len(gd), len(cd), 4)
        gd, cd = gd[-n:], cd[-n:]
        sg, sc = np.std(gd), np.std(cd)
        if sg < 1e-7 or sc < 1e-7:
            return 0.0
        corr = np.corrcoef(gd, cd)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(max(0.0, corr)) * CAUSAL_SCALE


class PickPlaceCombinedEnv(gym.Env):
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

        # Curriculum
        self._stage               = 1
        self._success_buf_adv     = deque(maxlen=ADVANCE_WINDOW)
        self._success_buf_reg     = deque(maxlen=REGRESS_WINDOW)
        self._stage_episode_count = 0

        # One-time flags
        self._contact_given    = False
        self._grasp_given      = False
        self._lift_given       = False
        self._close_near_given = False

        self._causal = GrangerCausalReward()

    # ── connection ──────────────────────────────────────────────────────────
    def _connect(self):
        if self._client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)

    # ── CSV ─────────────────────────────────────────────────────────────────
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

    def _log_step(self, step, stage, ee_d, ct_d, cz, ac, g, r, tr):
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            self._episode_count, step, stage,
            f"{ee_d:.4f}", f"{ct_d:.4f}", f"{cz:.4f}",
            int(ac), int(g), f"{self._gripper_state:.3f}",
            f"{r:.4f}", f"{tr:.4f}",
        ])

    # ── curriculum ──────────────────────────────────────────────────────────
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
                print(f"  [combined] → Stage {self._stage} (ep {self._episode_count})")

        if self._stage > 1:
            if (len(self._success_buf_reg) >= REGRESS_WINDOW
                    and self._stage_episode_count >= MIN_STAGE_EPISODES
                    and np.mean(self._success_buf_reg) < REGRESS_THRESHOLD):
                self._stage -= 1
                self._success_buf_adv.clear()
                self._success_buf_reg.clear()
                self._stage_episode_count = 0
                print(f"  [combined] ← Regress Stage {self._stage} (ep {self._episode_count})")

    def _stage_reset(self):
        tray_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)
        if self._stage == 1:
            ox = np.clip(tray_pos[0] + random.uniform(-0.12, 0.12), 0.35, 0.65)
            oy = np.clip(tray_pos[1] + random.uniform(-0.12, 0.12), -0.25, 0.25)
            p.resetBasePositionAndOrientation(
                self._object_id, [ox, oy, 0.02], [0,0,0,1],
                physicsClientId=self._client)
            joints = p.calculateInverseKinematics(
                self._robot, EE_LINK, [ox, oy, 0.15],
                restPoses=REST_POSE + [0.04, 0.04],
                physicsClientId=self._client)
            for j in ARM_JOINTS:
                p.resetJointState(self._robot, j, joints[j],
                                  physicsClientId=self._client)
        elif self._stage == 2:
            ox = random.uniform(0.35, 0.65)
            oy = random.uniform(-0.25, 0.25)
            p.resetBasePositionAndOrientation(
                self._object_id, [ox, oy, 0.02], [0,0,0,1],
                physicsClientId=self._client)
            joints = p.calculateInverseKinematics(
                self._robot, EE_LINK, [ox, oy, 0.18],
                restPoses=REST_POSE + [0.04, 0.04],
                physicsClientId=self._client)
            for j in ARM_JOINTS:
                p.resetJointState(self._robot, j, joints[j],
                                  physicsClientId=self._client)
        else:
            ox = random.uniform(0.35, 0.65)
            oy = random.uniform(-0.25, 0.25)
            p.resetBasePositionAndOrientation(
                self._object_id, [ox, oy, 0.02], [0,0,0,1],
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

        self._object_id = p.loadURDF(
            "cube_small.urdf", basePosition=[0.5, 0.0, 0.02],
            physicsClientId=self._client)
        self._container = p.loadURDF(
            "tray/traybox.urdf", basePosition=[0.5, 0.4, 0.0],
            physicsClientId=self._client)
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
        self._causal.reset()

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
            physicsClientId=self._client)
        for j in ARM_JOINTS:
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=200,
                physicsClientId=self._client)

        self._gripper_state = float(np.clip((gripper_cmd + 1.0) / 2.0, 0.0, 1.0))
        fp = FINGER_CLOSED + self._gripper_state * (FINGER_OPEN - FINGER_CLOSED)
        for fj in FINGER_JOINTS:
            p.setJointMotorControl2(
                self._robot, fj,
                controlMode=p.POSITION_CONTROL,
                targetPosition=fp, force=20,
                physicsClientId=self._client)

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

        self._causal.update(self._gripper_state, obj_pos)

        reward, terminated = self._compute_reward(
            ee_pos, obj_pos, tray_pos, cube_z, any_contact, grasp)
        self._total_reward += reward
        truncated = self._step_count >= MAX_STEPS

        if terminated or truncated:
            success = terminated and float(np.linalg.norm(obj_pos - tray_pos)) < PLACE_THRESHOLD
            self._update_curriculum(success)
            self._csv_file.flush()

        ee_d = float(np.linalg.norm(ee_pos  - obj_pos))
        ct_d = float(np.linalg.norm(obj_pos - tray_pos))
        self._log_step(self._step_count, self._stage, ee_d, ct_d,
                       cube_z, any_contact, grasp, reward, self._total_reward)

        return obs, reward, terminated, truncated, {
            "ee_cube_dist":   ee_d,
            "cube_tray_dist": ct_d,
            "cube_z":         cube_z,
            "any_contact":    any_contact,
            "grasp":          grasp,
            "gripper_state":  self._gripper_state,
            "stage":          self._stage,
        }

    # ── obs ──────────────────────────────────────────────────────────────────
    def _get_ee_pos(self):
        return list(p.getLinkState(self._robot, EE_LINK,
                                   physicsClientId=self._client)[0])

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
        any_contacts = p.getContactPoints(bodyA=self._robot, bodyB=self._object_id,
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
        obs = (arm_angles + arm_vels + ee_pos
               + list(obj_pos) + list(target_pos)
               + [cube_z, any_contact, grasp, self._gripper_state])
        return np.array(obs, dtype=np.float32)

    # ── reward ───────────────────────────────────────────────────────────────
    def _compute_reward(self, ee_pos, obj_pos, tray_pos,
                        cube_z, any_contact, grasp):
        dist_ee   = float(np.linalg.norm(ee_pos  - obj_pos))
        dist_tray = float(np.linalg.norm(obj_pos - tray_pos))
        r = 0.0

        # Base dense
        r -= 1.0 * dist_ee
        r -= 0.2 * dist_tray
        r -= 0.01

        # Dense gripper (Exp2)
        if dist_ee < GRIPPER_PROXIMITY:
            prox    = 1.0 - (dist_ee / GRIPPER_PROXIMITY)
            closure = 1.0 - self._gripper_state
            r      += 0.3 * prox * closure

        # Barrier B1: penalise closing far from cube
        closure_amt = 1.0 - self._gripper_state
        far_pen     = max(0.0, dist_ee - BARRIER_FAR_THRESHOLD)
        r          -= BARRIER_CLOSURE_FAR_COEF * closure_amt * far_pen

        # Barrier B2: penalise lift without grasp
        if not grasp:
            r -= BARRIER_PREMATURE_COEF * max(0.0, cube_z - 0.04)

        # Granger-causal
        r += self._causal.compute(dist_ee)

        # One-time
        if any_contact and not self._contact_given:
            r += 5.0; self._contact_given = True
        if dist_ee < GRIPPER_PROXIMITY and self._gripper_state < 0.4 \
                and not self._close_near_given:
            r += 8.0; self._close_near_given = True
        if grasp and not self._grasp_given:
            r += 15.0; self._grasp_given = True
        if cube_z > LIFT_THRESHOLD and not self._lift_given:
            r += 20.0; self._lift_given = True

        # Conditional transport
        if cube_z > LIFT_THRESHOLD:
            r += 0.2 * dist_tray
            r -= 0.8 * dist_tray

        # Terminal
        terminated = False
        if dist_tray < PLACE_THRESHOLD and cube_z > self._tray_z + 0.03:
            r += 100.0; terminated = True
        if cube_z < -0.10:
            r -= 10.0; terminated = True

        return r, terminated

    def render(self):
        pass

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = self._csv_writer = None
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None
