"""
pick_place_env.py  (v4 — anti-gaming, proper grasp, physics substeps)
----------------------------------------------------------------------
Changes from v3:
  - ALL bonuses are ONE-TIME per episode (no farming)
  - Grasp = BOTH finger links contacting cube + gripper closed (not arm bump)
  - 10 physics substeps per action (realistic timing: 500 steps = ~20s sim)
  - Success requires cube ABOVE tray rim (no pushing exploit)
  - Cube-off-table termination with penalty
  - Small time penalty (incentivise speed)
  - Weak tray distance signal always on (prevent "never lift" local optimum)

State (27-dim):
    arm_angles(7) + arm_velocities(7) + ee_pos(3) +
    obj_pos(3) + tray_pos(3) + cube_z(1) + any_contact(1) +
    grasp(1) + gripper_state(1)

Action (4-dim):
    [dx, dy, dz, gripper]
    dx,dy,dz ∈ [-0.05, 0.05] m per action (IK-resolved)
    gripper ∈ [-1, 1]  (+1 = fully open, -1 = fully closed)

Reward (one-time bonuses + dense shaping):
    EVERY STEP:
      -1.0 * dist(EE → cube)                           (reach)
      -0.2 * dist(cube → tray)                         (weak tray pull, always)
      -0.01                                             (time penalty)
    ONE-TIME:
      +5   first any-body contact with cube
      +15  proper grasp (both fingers + gripper closed)
      +20  first lift (cube_z > 0.05 m)
    CONDITIONAL (per-step, only when lifted):
      -1.0 * dist(cube → tray)                         (replaces weak pull)
    TERMINAL:
      +100 cube within 0.05m of tray AND cube_z > tray_z + 0.03
      -10  cube fell off table (cube_z < -0.10)
"""

import csv
import os
import random

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

# ── Franka Panda constants ─────────────────────────────────────────────────
ROBOT_URDF     = "franka_panda/panda.urdf"
ARM_JOINTS     = list(range(7))
FINGER_JOINTS  = [9, 10]
FINGER_LINKS   = [9, 10]           # link index = joint index in PyBullet
EE_LINK        = 11
FINGER_OPEN    = 0.04
FINGER_CLOSED  = 0.00
REST_POSE      = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# ── task constants ──────────────────────────────────────────────────────────
PLACE_THRESHOLD = 0.05              # tightened from 0.10
LIFT_THRESHOLD  = 0.05
MAX_STEPS       = 500
DELTA_CLIP      = 0.05
N_SUBSTEPS      = 10                # physics steps per action
WORKSPACE = {
    "low":  np.array([0.25, -0.55, 0.01]),
    "high": np.array([0.80,  0.55, 0.70]),
}


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, log_dir="logs/diagnostics"):
        super().__init__()
        self.render_mode = render_mode
        self.log_dir     = log_dir
        self._client     = None

        # 4D: [dx, dy, dz, gripper]
        self.action_space = spaces.Box(
            low=np.array([-DELTA_CLIP, -DELTA_CLIP, -DELTA_CLIP, -1.0],
                         dtype=np.float32),
            high=np.array([DELTA_CLIP, DELTA_CLIP, DELTA_CLIP, 1.0],
                          dtype=np.float32),
        )

        # 27-dim observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

        self._step_count    = 0
        self._episode_count = 0
        self._total_reward  = 0.0
        self._gripper_state = 1.0
        self._csv_writer    = None
        self._csv_file      = None

        # One-time bonus flags (reset every episode)
        self._contact_given = False
        self._grasp_given   = False
        self._lift_given    = False

    # ── connection ─────────────────────────────────────────────────────────
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

    # ── CSV diagnostics ────────────────────────────────────────────────────
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
                "reward", "total_reward",
            ])

    def _log_step(self, step, ee_cube_dist, cube_tray_dist,
                  cube_z, any_contact, grasp, reward, total_reward):
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            self._episode_count, step,
            f"{ee_cube_dist:.4f}", f"{cube_tray_dist:.4f}",
            f"{cube_z:.4f}", int(any_contact), int(grasp),
            f"{self._gripper_state:.3f}",
            f"{reward:.4f}", f"{total_reward:.4f}",
        ])

    # ── reset ──────────────────────────────────────────────────────────────
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

        # Arm → rest pose
        for i, angle in zip(ARM_JOINTS, REST_POSE):
            p.resetJointState(self._robot, i, angle,
                              physicsClientId=self._client)

        # Gripper → open
        for fj in FINGER_JOINTS:
            p.resetJointState(self._robot, fj, FINGER_OPEN,
                              physicsClientId=self._client)
        self._gripper_state = 1.0

        # Randomise cube
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

        # Get tray height for success check
        tray_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)
        self._tray_z = tray_pos[2]

        self._step_count    = 0
        self._episode_count += 1
        self._total_reward  = 0.0

        # Reset one-time flags
        self._contact_given = False
        self._grasp_given   = False
        self._lift_given    = False

        # Warm-up physics
        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)

        return self._get_obs(), {}

    # ── step ───────────────────────────────────────────────────────────────
    def step(self, action):
        delta_ee    = action[:3]
        gripper_cmd = float(action[3])

        # ── EE control via IK ──
        ee_pos    = np.array(self._get_ee_pos())
        target_ee = np.clip(ee_pos + delta_ee,
                            WORKSPACE["low"], WORKSPACE["high"])
        joint_poses = p.calculateInverseKinematics(
            self._robot, EE_LINK, target_ee.tolist(),
            restPoses=REST_POSE + [0.04, 0.04],  # arm + fingers rest
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

        # ── Gripper control ──
        self._gripper_state = float(
            np.clip((gripper_cmd + 1.0) / 2.0, 0.0, 1.0)
        )
        finger_pos = (FINGER_CLOSED
                      + self._gripper_state * (FINGER_OPEN - FINGER_CLOSED))
        for fj in FINGER_JOINTS:
            p.setJointMotorControl2(
                self._robot, fj,
                controlMode=p.POSITION_CONTROL,
                targetPosition=finger_pos,
                force=20,
                physicsClientId=self._client,
            )

        # ── Physics substeps (10 steps = ~0.042 s sim time per action) ──
        for _ in range(N_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._client)

        self._step_count += 1

        # ── Observe ──
        obs = self._get_obs()
        ee_pos   = obs[14:17]
        obj_pos  = obs[17:20]
        tray_pos = obs[20:23]
        cube_z   = float(obs[23])
        any_contact = bool(obs[24])
        grasp       = bool(obs[25])

        # ── Reward ──
        reward, terminated = self._compute_reward(
            ee_pos, obj_pos, tray_pos, cube_z, any_contact, grasp
        )
        self._total_reward += reward
        truncated = self._step_count >= MAX_STEPS

        ee_cube_dist   = float(np.linalg.norm(ee_pos  - obj_pos))
        cube_tray_dist = float(np.linalg.norm(obj_pos - tray_pos))
        self._log_step(
            self._step_count, ee_cube_dist, cube_tray_dist,
            cube_z, any_contact, grasp, reward, self._total_reward,
        )

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

    # ── observation ────────────────────────────────────────────────────────
    def _get_ee_pos(self):
        ls = p.getLinkState(self._robot, EE_LINK,
                            physicsClientId=self._client)
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

        # Any contact: any link of robot touching cube
        any_contacts = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            physicsClientId=self._client,
        )
        any_contact = float(len(any_contacts) > 0)

        # Proper grasp: BOTH finger links contacting cube + gripper closed
        c_finger1 = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            linkIndexA=FINGER_LINKS[0],
            physicsClientId=self._client,
        )
        c_finger2 = p.getContactPoints(
            bodyA=self._robot, bodyB=self._object_id,
            linkIndexA=FINGER_LINKS[1],
            physicsClientId=self._client,
        )
        grasp = float(
            len(c_finger1) > 0
            and len(c_finger2) > 0
            and self._gripper_state < 0.5     # fingers mostly closed
        )

        obs = (arm_angles + arm_vels
               + ee_pos + list(obj_pos) + list(target_pos)
               + [cube_z, any_contact, grasp, self._gripper_state])
        return np.array(obs, dtype=np.float32)

    # ── reward (hardened against gaming) ───────────────────────────────────
    def _compute_reward(self, ee_pos, obj_pos, tray_pos,
                        cube_z, any_contact, grasp):
        dist_ee_obj   = np.linalg.norm(ee_pos  - obj_pos)
        dist_obj_tray = np.linalg.norm(obj_pos - tray_pos)

        reward = 0.0

        # ── Dense (every step) ─────────────────────────────────────────
        # Reach: always pull EE toward cube
        reward -= 1.0 * dist_ee_obj

        # Weak tray pull: always on, prevents "never lift" local optimum
        reward -= 0.2 * dist_obj_tray

        # Time penalty: incentivise solving quickly
        reward -= 0.01

        # ── One-time bonuses ───────────────────────────────────────────
        # First contact (any body part)
        if any_contact and not self._contact_given:
            reward += 5.0
            self._contact_given = True

        # Proper grasp (both fingers + closed gripper)
        if grasp and not self._grasp_given:
            reward += 15.0
            self._grasp_given = True

        # First lift
        if cube_z > LIFT_THRESHOLD and not self._lift_given:
            reward += 20.0
            self._lift_given = True

        # ── Conditional dense (only when lifted) ──────────────────────
        if cube_z > LIFT_THRESHOLD:
            # Stronger tray pull replaces the weak one
            reward += 0.2 * dist_obj_tray      # undo weak pull
            reward -= 1.0 * dist_obj_tray       # full strength pull

        # ── Terminal ───────────────────────────────────────────────────
        terminated = False

        # Success: cube close to tray AND elevated (no pushing exploit)
        if dist_obj_tray < PLACE_THRESHOLD and cube_z > self._tray_z + 0.03:
            reward    += 100.0
            terminated = True

        # Failure: cube fell off table
        if cube_z < -0.10:
            reward    -= 10.0
            terminated = True

        return reward, terminated

    # ── render / close ─────────────────────────────────────────────────────
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
