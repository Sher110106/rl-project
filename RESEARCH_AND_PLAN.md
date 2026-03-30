# Deep Research Report & Novel Implementation Plan
## RL Pick-and-Place: Beyond Standard Approaches

**Prepared:** 2026-03-26
**Context:** PyBullet Franka Panda, 27-dim state, 4-dim action, SAC/SB3 baseline
**Research scope:** Robotics RL, control theory, information theory, neuroscience, physics, mathematics, music theory, statistical physics

---

## Part 1: State of the Project

### What Has Been Built

A full RL pipeline on a Franka Panda pick-and-place task (PyBullet, v4):

| Dimension | Detail |
|---|---|
| **State (27-dim)** | 7 joint angles + 7 joint velocities + EE pos (3) + cube pos (3) + tray pos (3) + cube_z + contact + grasp + gripper_state |
| **Action (4-dim)** | [dx, dy, dz, gripper] — EE delta via IK + gripper open/close |
| **Algorithm** | SAC (Stable-Baselines3) |
| **Reward** | Anti-gaming: one-time bonuses + dense reach/tray pull |

### What Has Been Tried

| Experiment | Best Result | Problem |
|---|---|---|
| v1 SAC baseline (100k) | -416 reward, 0% success | Only reaches, never grasps |
| v4 SAC anti-gaming (1.5M) | -51.5 reward, 0% success | Honest signal, still can't grasp |
| **EXP1: HER + SAC (500k)** | -500 reward, **0% success** | Sparse reward too hard, no seed trajectory |
| **EXP2: Dense Gripper Shaping (1M)** | **+9.62 reward** | Consistent reaching+contact, **no full task completion** |
| **EXP3: Curriculum SAC (1M)** | -49.3 reward, **11% success** | Unstable: regresses after stage advance |

### The Core Bottleneck

Three distinct failure modes, each needing a different solution:

1. **Reach → Grasp gap** (grip closure timing + spatial alignment) — solved partially by Exp2
2. **Grasp → Lift gap** (maintaining grip while applying upward force against gravity) — unsolved
3. **Lift → Place gap** (transporting to tray without dropping) — unsolved, only Exp3 touched it

The agent understands *where* to go but cannot chain the complete manipulation sequence reliably.

---

## Part 2: Cross-Domain Research Synthesis

### Field 1: Control Theory — Barrier Functions & Lyapunov Methods

**Key finding:** Control barrier functions (CBFs) are used in safe robotics control to define "safe sets" — regions of state space the system must remain within. A barrier function B(s) ≥ 0 inside the safe set, and its time derivative must satisfy Ḃ ≥ -αB(s) (ensuring the system doesn't escape).

**IISc paper (2024):** Applied BF-inspired reward shaping to locomotion — achieved 1.4-2.8× faster convergence, 50% less actuation effort. Key: the barrier encodes "stay within desired behavioral limits" as a reward signal.

**For pick-and-place, untried idea:** Instead of using barriers for safety, use them as **sequential phase gates**. Define:
- B₁(s): "EE must be within 0.1m of cube before gripper closes" — barrier collapses to zero when violated
- B₂(s): "Gripper must be closed before lift force applied"
- B₃(s): "Cube must be lifted before transport begins"

The shaped reward becomes: r_shaped = r_base + α₁·Ḃ₁ + α₂·Ḃ₂ + α₃·Ḃ₃

**Why this hasn't been done for pick-place:** All barrier function RL papers apply to locomotion safety. Using barriers as *task-sequencing gates* for dexterous manipulation is new.

---

### Field 2: Information Theory — Granger Causality & Mutual Information

**Key finding (HIntS, 2023):** Granger-Causal Hierarchical Skill Discovery uses Granger causality to detect when robot actions causally influence specific objects. An "interaction skill" exists when past gripper actions Granger-cause future changes in object state.

**Mathematical definition:** Action sequence A_{t-k:t} Granger-causes object state O_{t:t+k} if:
```
Var(O_{t+1} | O_t, A_t) < Var(O_{t+1} | O_t)
```
i.e., knowing the action reduces prediction error for the object's future state.

**Novel application for pick-and-place:** Define a **Granger-Causal Intrinsic Reward**:
```python
r_causal = MI(gripper_action_{t-5:t}; delta_cube_pos_{t:t+5})
```
Estimate this online using a small neural network trained to predict cube movement from gripper history. The reward is high when the robot is *actually manipulating* the cube (versus just being nearby). This directly measures manipulation quality, not just spatial proximity.

**Why this is novel:** No paper has used Granger causality as a *continuous dense reward signal* for pick-and-place. HIntS uses it for skill discovery (discrete), not as a training reward (continuous).

---

### Field 3: Statistical Physics — Phase Transitions & Order Parameters

**Key insight:** The pick-place task exhibits distinct *phases* analogous to physical phase transitions:

| Phase | Order Parameters | Signature |
|---|---|---|
| **Free (searching)** | EE-cube dist > 0.08 | Random exploration |
| **Contact** | contact=True, dist < 0.05 | First contact event |
| **Grasping** | grasp=True | Both fingers on cube |
| **Lifting** | cube_z > 0.05 | Cube elevated |
| **Transporting** | cube lifted, EE moving toward tray | Motion phase |
| **Placing** | cube-tray dist < 0.05 | Final placement |

**Novel reward formulation:** Model transitions between phases as rewards, not arbitrary distance thresholds. Define phase membership functions φ_i(s) ∈ [0,1] (soft indicators). The reward for a *phase transition* (φ_i → φ_{i+1}) is far richer than a fixed +15 bonus.

**From statistical mechanics:** Use the concept of an "order parameter" that continuously measures how far through the phase transition the system is. For the grasp phase transition, the order parameter is:
```
ψ_grasp = (1 - finger_aperture/max_aperture) * exp(-dist_EE_cube / σ)
```
This is a continuous, smooth measure of "how grasped" the cube is — much richer than binary contact detection.

---

### Field 4: Neuroscience — Central Pattern Generators & Motor Primitives

**Key finding:** Central Pattern Generators (CPGs) are neural circuits in the spinal cord that generate rhythmic motor patterns (walking, breathing, swimming). Recent work (Nature 2025, arxiv 2305.07300) shows RL can modulate CPG parameters to achieve locomotion tasks, achieving much better sample efficiency than direct joint control.

**The insight for manipulation:** Grasping and manipulation also have a natural rhythmic structure — approach, orient, close, squeeze, lift. Biological robots (hands) use CPG-like coordination patterns, not independent joint control.

**Novel architecture for pick-place:**

Define a **Manipulation CPG** with oscillatory primitives:
```
gripper_aperture(t) = A * sin(ωt + φ) + offset
EE_approach(t) = sigmoid(-k * dist_EE_cube + b) * velocity
```

RL doesn't control joints directly. Instead, RL learns to modulate:
- **A** (amplitude): how aggressively to close gripper
- **ω** (frequency): how fast to oscillate
- **φ** (phase): when in the cycle to attempt contact
- **k** (approach gain): how fast to approach

This **dramatically reduces effective action dimensionality** from 4-continuous to 4-scalar modulations of a structured motor program.

**Why this is novel for pick-place:** CPG + RL has been done only for locomotion. For manipulation, there are Dynamic Movement Primitives (DMPs), but a true oscillatory CPG for grasping with RL modulation has not been done.

---

### Field 5: Optimal Transport — Wasserstein Distance as Reward

**Key finding (OTPR, 2025):** Score-Based Diffusion Policy with Optimal Transport Reward uses Wasserstein distance between the agent's trajectory distribution and demonstration trajectories as a reward signal. Achieves state-of-the-art on manipulation benchmarks.

**The approach:**
1. Collect a small set of successful demonstrations (from Exp3's 11% success episodes)
2. For each episode, compute the Wasserstein distance W(τ_agent, τ_demo) between trajectory distributions
3. Use this as a shaped reward: r_ot = -W(τ_agent, τ_demo) + r_base

**Novel adaptation for our setup:**
- Use the **existing successful episodes from Exp3** as demonstrations (don't need human demos)
- Compute OT distance in a *reduced state space*: (EE_pos, cube_pos, gripper_state) rather than full 27-dim
- This lets a nearly-successful partial trajectory get credit for being close to the demonstration distribution

**Why this is feasible:** We already have successful episodes from Exp3's 11% success rate. These become demonstrations for OT-based reward shaping. This is **self-improving** — as the agent succeeds more, the demonstration set grows richer.

---

### Field 6: Model-Based RL — Potential Landscape Shaping (SLOPE)

**Key finding (SLOPE, 2026):** In sparse reward MBRL, fitting scalar rewards creates a "gradient-free landscape" where the planner gets no directional signal. SLOPE uses distributional value modeling to estimate an optimistic upper-bound Q-value, then uses this as a potential function:
```
r_shaped = r_sparse + [Q_UCB(s') - Q_UCB(s)]
```
where Q_UCB is the upper confidence bound from a distributional Q-network (e.g., quantile regression).

**The key property:** This is a valid potential-based reward shaping (PBRS) — it doesn't change the optimal policy (Ng et al., 1999 theorem), but it provides a smooth gradient landscape.

**Novel combination for our task:** Use distributional SAC (SAC with quantile regression critic) and extract Q_UCB as potential:
```python
# In the replay buffer update:
q_values = [critic_i(s, a) for critic_i in critics]  # ensemble of quantile critics
q_ucb = mean(q_values) + β * std(q_values)  # UCB estimate
r_shaped = r_sparse + γ * q_ucb(s') - q_ucb(s)
```

**Why this hasn't been done for pick-place:** SLOPE applies this in MBRL (TD-MPC2, DreamerV3). Applying the same potential-based UCB shaping to model-free SAC for robotic manipulation is new.

---

### Field 7: Replay Buffer — Contact Energy Based Prioritization (CEBP)

**Key finding (TUM, 2024):** Contact Energy Based Prioritization (CEBP) modifies HER by prioritizing replays where contact force was high and object displacement was large. Standard HER uniformly samples from the replay buffer, but contact-rich experiences provide the most information for manipulation learning.

**The formula:**
```
priority(τ) = λ₁ * max_contact_force(τ) + λ₂ * total_object_displacement(τ)
```

**Specific novelty for our setup:** Our environment already tracks `any_contact`, `grasp`, and object positions per step in the diagnostic CSV. We can compute:
```python
priority = (
    λ₁ * contact_bonus_received  # +5 or +15 given in episode
  + λ₂ * total_cube_displacement  # sum of |Δcube_pos| over episode
  + λ₃ * max_cube_z               # highest point the cube reached
)
```

The replay buffer samples from this distribution rather than uniformly. This doesn't require new infrastructure — just modify the SB3 replay buffer's sampling weights.

**Novel extension beyond CEBP:** Add a *temporal decay* to prioritization: recent contact-rich transitions are prioritized more, simulating working memory for manipulation. No paper has done this.

---

### Field 8: Representation Learning — Causal Disentanglement

**Key finding (2024-2025):** Causal disentangled representations explicitly separate state factors by their causal roles. For robot manipulation: (a) robot-controlled factors, (b) object state, (c) fixed scene elements.

**Novel architecture for pick-place:**

Augment the SAC policy with an **auxiliary causal disentanglement loss**:
```
state_encoder(obs) → [z_robot, z_object, z_scene]

# Auxiliary losses:
L_indep = MI(z_scene, z_robot)  # scene shouldn't change with robot actions
L_control = MI(action, Δz_object)  # actions should cause object change
L_recon = ||decode([z_robot, z_object, z_scene]) - obs||²
```

The policy π(a|z_robot, z_object, z_scene) benefits from this structure because:
- z_scene (tray position) is constant → compress it
- z_object (cube) should respond to z_robot (gripper) during grasp
- This structure makes the policy more **generalizable** to new cube/tray positions

---

### Field 9: Hierarchical RL — Diffusion Meets Options

**Key finding (2024):** "Diffusion Meets Options" uses diffusion models to represent options (skills) within a hierarchical RL framework. High-level policy selects options; each option is a diffusion model generating action sequences.

**Novel 3-tier architecture for pick-place:**

```
Tier 1 (Manager, low frequency): Selects skill from {REACH, GRASP, LIFT, TRANSPORT, PLACE}
Tier 2 (Skill, medium frequency): Diffusion model for each skill generates 10-step action sequences
Tier 3 (Executor, high frequency): Per-step physics execution
```

**Why diffusion for skills:** Diffusion models naturally capture *multimodal action distributions* — for grasping, there are multiple valid grip orientations. A standard Gaussian policy would average these into an invalid middle position.

**How to bootstrap:** Use Exp3's successful episodes to pre-train each skill's diffusion model. The Manager is trained online with RL. This leverages existing successful trajectories.

---

### Field 10: Truly Cross-Domain — Reservoir Computing for Contact Dynamics

**Key finding:** Echo State Networks (ESNs) are randomly fixed recurrent neural networks where only the linear readout layer is trained. They excel at modeling nonlinear dynamical systems because the random reservoir creates a rich nonlinear feature space.

**The connection to pick-and-place:** Contact dynamics (gripper-cube interaction) are highly nonlinear, stochastic, and history-dependent — exactly the domain where ESNs excel. Standard MLP forward models in MBRL struggle here.

**Novel idea:** Use an ESN as the forward dynamics model specifically for the **contact phase**:
```
# Normal regime (no contact): standard MLP dynamics model
# Contact regime (any_contact=True): switch to ESN dynamics model

s_{t+1} = ESN(s_t, a_t, reservoir_state)
```

The ESN can capture the history-dependent nature of gripping (whether fingers are sliding vs. stable) without training a full recurrent network. Only the readout weights (100-500 params) need training.

**Novel combination:** No paper has used ESN specifically for the contact/manipulation phase in a model-based RL hybrid. This is inspired by reservoir computing's success in nonlinear dynamical systems but applied to a new domain.

---

### Field 11: Active Inference — Free Energy Minimization as RL Replacement

**Key finding (Friston FEP, 2023; Nature 2025):** Active inference agents minimize "free energy" (a variational bound on surprise) rather than maximizing reward. The agent maintains a generative model of the world and takes actions to make observations match predictions.

**For pick-and-place:** Define the agent's "desired state distribution" as:
```
P_desired(s) = N(s_goal, σ²)  # Gaussian around successful placement state
```

The free energy to minimize:
```
F = KL[Q(s) || P(s)] + expected_surprise
  = KL[Q(s) || P_desired(s)] + E[-log P(o|s)]
```

Actions minimize the divergence between the agent's current state distribution and the desired distribution. This **eliminates the need for a hand-crafted reward function entirely**.

**Why interesting for our task:** Our current reward struggles with phase transitions (the agent can get stuck in local optima). Active inference naturally avoids this through belief propagation across the full trajectory.

---

### Field 12: Self-Improvement — On-Manifold Exploration (SOE)

**Key finding (2025):** Self-Improvement via On-Manifold Exploration (SOE) learns a compact latent representation of task-relevant factors, then constrains exploration to the *manifold of valid actions*, ensuring safe, efficient exploration.

**For our task:** The "valid action manifold" for pick-and-place is a low-dimensional subspace of the 4D action space. Most random actions are useless; only those that move EE toward the cube, or close the gripper near the cube, are productive.

**Novel implementation:**
1. Train a variational autoencoder (VAE) on successful trajectories from Exp3
2. The latent space z defines the "valid manipulation manifold"
3. Add a manifold-projection regularization to the SAC policy:
   ```
   L_policy = L_SAC + λ * ||a - decode(encode(a))||²
   ```
4. The policy is encouraged to output actions that lie on the valid manipulation manifold

---

## Part 3: The Most Novel Proposals (Ranked)

### Rank 1: Granger-Causal Dense Reward (MOST NOVEL)

**Never done for pick-and-place in any paper found.**

The idea: reward the robot for *causally influencing* the cube, not just for being near it.

```python
# Online estimation of causal influence
class GrangerCausalReward:
    def __init__(self, window=10):
        # Small network to predict delta_cube_pos from gripper history
        self.causal_model = MLP([gripper_action * window + cube_pos, 64, 64, 3])
        self.baseline_model = MLP([cube_pos, 64, 3])  # no gripper info

    def compute(self, gripper_history, cube_pos_history):
        # Granger causality: does gripper reduce prediction error?
        pred_with_gripper = self.causal_model(gripper_history, cube_pos_history[-1])
        pred_without_gripper = self.baseline_model(cube_pos_history[-1])

        causal_influence = max(0,
            MSE(baseline, actual) - MSE(causal_model, actual)
        )
        return causal_influence  # reward signal

r_total = r_base + α * r_causal
```

**Intuition:** When the gripper is far from the cube, knowing gripper actions doesn't help predict cube movement — causal reward is near zero. When grasping, gripper actions *determine* cube movement — causal reward spikes. This provides a smooth, dense, semantically correct reward for manipulation.

**Expected improvement:** Directly solves the reach→grasp→lift chain by rewarding *effective* interaction at each sub-goal.

---

### Rank 2: Distributional SAC with Q-UCB Potential Shaping (SLOPE for Model-Free)

**Key insight:** Our task has sparse task-completion reward (+100). Even with dense shaping, the gradient landscape is flat until the agent accidentally succeeds. SLOPE showed that using the Q-function's *uncertainty* as a potential dramatically helps.

**Implementation:**
```python
# Distributional SAC: critic predicts Q-value distribution, not just mean
# Using Quantile Regression SAC (QR-SAC):

class QuantileSAC(SAC):
    def __init__(self, n_quantiles=51, ...):
        # Critic predicts distribution of Q-values via quantile regression
        ...

    def compute_potential(self, obs, action):
        quantiles = self.critic.forward_quantiles(obs, action)
        # Upper confidence bound: mean + β * std
        q_ucb = quantiles.mean() + self.beta * quantiles.std()
        return q_ucb

# Shaped reward in replay buffer update:
r_shaped = r_sparse + γ * Q_UCB(s') - Q_UCB(s)
```

**Why model-free (not MBRL):** Our PyBullet setup can run plenty of real steps; we don't need a world model. The potential shaping provides gradient signal without requiring a world model.

---

### Rank 3: Phase-Ordered Barrier Function Reward

**Combines control theory (CBFs) with task structure.**

Define a phase-sequencing reward that is provably policy-invariant (via PBRS theorem):

```python
def phase_barrier_reward(obs, prev_obs):
    ee_pos, cube_pos, cube_z = obs[14:17], obs[17:20], obs[23]
    gripper_state = obs[26]

    dist_ee_cube = np.linalg.norm(ee_pos - cube_pos)

    # Phase 1 barrier: gripper should close ONLY when near cube
    # B1(s) > 0 when gripper is opening and EE is far from cube (bad)
    # Penalize closing gripper when far away (wasted action)
    b1 = max(0, (1 - gripper_state) * (dist_ee_cube - 0.08))

    # Phase 2 barrier: should not lift until grasped
    grasp = obs[25]
    b2 = max(0, (cube_z - 0.04) * (1 - grasp))

    # Phase 3 barrier: cube should not move laterally when lifting
    if cube_z > 0.04:
        cube_xy_drift = np.linalg.norm(cube_pos[:2] - tray_pos[:2] - stable_lift_direction)
        b3 = cube_xy_drift * grasp  # penalize erratic transport

    # Shaped reward (potential-based, policy-invariant)
    # F(s) = -α1*B1 - α2*B2 - α3*B3 (Lyapunov potential)
    F_now = -0.5*b1 - 1.0*b2 - 0.3*b3
    F_prev = compute_F(prev_obs)
    r_barrier = γ * F_now - F_prev

    return r_barrier
```

**What this does:** Creates a smooth "channel" through the task phases. The barrier functions penalize "out-of-phase" behaviors — closing the gripper while far from the cube, trying to lift without a grasp, etc.

---

### Rank 4: Contact-Energy HER with Temporal Priority Decay

**CEBP (TUM 2024) + temporal decay novelty.**

```python
class ContactPrioritizedHER(HerReplayBuffer):
    def compute_priority(self, episode):
        # Contact energy component
        contact_bonus = episode.info['contact_bonus']  # 5 or 15 if earned

        # Object displacement component
        cube_positions = episode.observations[:, 17:20]
        total_displacement = np.sum(np.linalg.norm(np.diff(cube_positions, axis=0), axis=1))

        # Maximum lift achieved
        max_cube_z = np.max(episode.observations[:, 23])

        # Temporal recency weight (novel addition)
        recency_weight = np.exp(-self.age[episode.id] / self.decay_tau)

        priority = (
            2.0 * contact_bonus +
            1.0 * total_displacement +
            3.0 * max_cube_z +
            recency_weight
        )
        return priority
```

**Why temporal decay is novel:** Contact-rich experiences from early training (when the robot was learning to reach) are less informative than recent contact-rich experiences (when the robot is learning to grasp). No existing paper adds temporal decay to contact-based replay prioritization.

---

### Rank 5: Self-Bootstrapped Optimal Transport Reward

**Use our own successful trajectories from Exp3 as demonstrations.**

```python
class OTReward:
    def __init__(self, demo_trajectories):
        # Use successful episodes from Exp3 as demonstrations
        # State space reduced to (EE_pos, cube_pos, gripper_state) — 7D
        self.demos = [traj[:, [14,15,16, 17,18,19, 26]] for traj in demo_trajectories]

    def compute(self, agent_trajectory):
        agent_traj = agent_trajectory[:, [14,15,16, 17,18,19, 26]]

        # Compute min Wasserstein distance to any demonstration
        min_ot = min(
            wasserstein_distance_1d(agent_traj, demo)
            for demo in self.demos
        )

        # Shaped reward: negative OT distance (minimize distance to demos)
        return -0.1 * min_ot

    def update_demos(self, new_successful_traj):
        # Self-improvement: add new successful episodes
        self.demos.append(new_successful_traj)
        if len(self.demos) > 100:
            self.demos.pop(0)  # keep recent demos
```

**Self-bootstrapping property:** As the agent achieves more successes (even occasional ones from the existing curriculum model's 11%), the demonstration set grows richer. This creates a virtuous cycle.

---

### Rank 6: CPG-Modulated Gripper Architecture

**Inspired by biological motor control. Novel for manipulation.**

```python
class CPGGripperWrapper(gym.Wrapper):
    """
    Instead of directly outputting gripper position,
    the policy outputs CPG modulation parameters.
    The CPG generates smooth gripper trajectories.
    """
    def __init__(self, env):
        super().__init__(env)
        self.phi = 0.0  # CPG phase
        self.A = 0.04   # amplitude (finger aperture range)
        self.omega = 2.0  # frequency

        # New action space: [dx, dy, dz, cpg_amplitude, cpg_freq, cpg_phase_shift]
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0.0, 0.1, -np.pi]),
            high=np.array([0.05, 0.05, 0.05, 0.04, 5.0, np.pi]),
        )

    def step(self, action):
        dx, dy, dz = action[:3]
        self.A = action[3]
        self.omega = action[4]
        phase_shift = action[5]

        # CPG generates smooth gripper oscillation
        self.phi += self.omega * 0.05  # dt = 0.05s
        gripper_cmd = self.A * np.sin(self.phi + phase_shift)

        # Map to original env action
        base_action = np.array([dx, dy, dz, gripper_cmd])
        return self.env.step(base_action)
```

**Why this helps:** Standard SAC produces noisy, uncorrelated gripper commands. A CPG generates temporally smooth, coordinated gripper sequences that are far more likely to produce stable grasps. The RL policy learns the "macro-parameters" of the grasping rhythm, not the micro-level noise.

---

## Part 4: Grand Unified Novel Architecture

Combining the best ideas into a single coherent system:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NOVEL PICK-PLACE AGENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  OBSERVATION PIPELINE:                                           │
│  raw_obs (27D) → Causal Encoder → [z_robot(8), z_cube(6), z_tray(4)] │
│                                                                   │
│  POLICY (SAC with Distributional Critic):                        │
│  [z_robot, z_cube, z_tray] → Transformer → action chunks (5-step) │
│                                                                   │
│  ACTION EXECUTION:                                               │
│  action_chunks → CPG Module → smooth joint commands              │
│                                                                   │
│  REWARD SIGNALS (additive):                                      │
│  r = r_base (existing)                                           │
│    + r_granger (causal influence on cube)                        │
│    + r_barrier (phase-gate compliance)                           │
│    + r_potential (Q-UCB shaping)                                 │
│    + r_ot (OT distance to successful demos)                      │
│                                                                   │
│  REPLAY BUFFER:                                                  │
│  Contact-Energy HER with Temporal Decay Priority                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Prioritized Implementation Plan

This plan is sequenced by **expected impact × implementation feasibility**. Each experiment builds on the previous one.

---

### Phase A: Foundation Fixes (Week 1)
*Before novel ideas, fix known issues with current setup*

#### A1: HER with Dense Reward (Combine Exp1 + Exp2)
**Problem:** Exp1 (HER) failed due to pure sparse reward. Exp2 (dense shaping) gave good signal but no task completion.
**Fix:** Enable HER on top of Exp2's dense reward.
```python
# In train.py: add HER wrapper
from stable_baselines3.her import HerReplayBuffer
model = SAC(
    "MultiInputPolicy",  # Dict obs for HER
    env=goal_conditioned_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    **SAC_KWARGS
)
```
**Expected:** HER now has seed trajectories from the dense reward signal. Expected: 10-20% success rate.

#### A2: Curriculum Stability Fix
**Problem:** Exp3 succeeds at Stage 1 (20%) but regresses at Stage 3.
**Fix:** Slow curriculum advancement threshold + performance-based regression:
```python
# Current: advance if success_rate > 15% over last 100 episodes
# New: advance if success_rate > 30% sustained for 300 episodes
#      regress if success_rate < 5% for 200 episodes
ADVANCE_THRESHOLD = 0.30  # was 0.15
ADVANCE_WINDOW = 300       # was 100
REGRESS_THRESHOLD = 0.05
REGRESS_WINDOW = 200
```
**Expected:** Stabilize at Stage 2-3, push success rate to 20-30%.

---

### Phase B: Novel Reward Engineering (Week 2)
*Add novel reward components one at a time to measure individual contribution*

#### B1: Phase-Ordered Barrier Function Reward
**Effort:** Low (pure reward modification, no architecture change)
**Files to modify:** `pick_place_env.py` — `_compute_reward()`
**Implementation:**
```python
def _compute_barrier_reward(self, ee_pos, obj_pos, cube_z, grasp):
    dist_ee_cube = np.linalg.norm(ee_pos - obj_pos)

    # B1: Gripper should only close when near cube
    # Penalize closing when far away
    closure_far = (1.0 - self._gripper_state) * max(0, dist_ee_cube - 0.1)

    # B2: Only lift when grasped
    premature_lift = max(0, cube_z - 0.03) * (1.0 - grasp)

    r_barrier = -0.5 * closure_far - 2.0 * premature_lift
    return r_barrier
```
**Why now:** Pure reward addition, 0 architecture changes. If this helps, it's the most efficient improvement possible.
**Expected improvement:** 15-25% faster convergence based on similar papers.

#### B2: Granger-Causal Intrinsic Reward (Simplified)
**Effort:** Medium (new module, online training)
**A simplified version that doesn't require a second neural network:**
```python
class SimpleCausalReward:
    def __init__(self, window=8):
        self.gripper_history = deque(maxlen=window)
        self.cube_history = deque(maxlen=window)

    def update(self, gripper_state, cube_pos):
        self.gripper_history.append(gripper_state)
        self.cube_history.append(cube_pos.copy())

    def compute(self):
        if len(self.gripper_history) < 4:
            return 0.0

        # Simplified Granger test: correlation between gripper change and cube change
        gripper_changes = np.diff(list(self.gripper_history))
        cube_movements = np.linalg.norm(np.diff(list(self.cube_history), axis=0), axis=1)

        # High correlation = causal influence = reward
        if np.std(gripper_changes) < 1e-6:
            return 0.0
        correlation = np.corrcoef(gripper_changes[-4:], cube_movements[-4:])[0,1]
        return max(0, correlation) * 0.5  # scale factor
```
**Expected:** Provides dense signal during grasp phase. Novel contribution if published.

---

### Phase C: Replay Buffer Improvements (Week 2-3)

#### C1: Contact Energy Based Prioritization (CEBP-TD)
**Combines TUM 2024 paper with our novel temporal decay**
```python
class ContactPriorityBuffer(ReplayBuffer):
    def __init__(self, *args, decay_tau=50000, **kwargs):
        super().__init__(*args, **kwargs)
        self.priorities = np.ones(self.buffer_size)
        self.timestamps = np.zeros(self.buffer_size)
        self.decay_tau = decay_tau
        self.current_step = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        # Compute priority based on contact energy
        contact = infos[0].get('any_contact', False)
        cube_z = infos[0].get('cube_z', 0)

        base_priority = 1.0
        if contact:
            base_priority += 2.0
        if cube_z > 0.05:
            base_priority += 3.0

        self.priorities[self.pos] = base_priority
        self.timestamps[self.pos] = self.current_step
        self.current_step += 1

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env=None):
        # Apply temporal decay to priorities
        age = self.current_step - self.timestamps[:self.n_envs * self.pos]
        decay = np.exp(-age / self.decay_tau)
        effective_priority = self.priorities[:len(decay)] * decay

        probs = effective_priority / effective_priority.sum()
        indices = np.random.choice(len(probs), size=batch_size, p=probs)
        return self._get_samples(indices, env=env)
```

---

### Phase D: Architecture-Level Changes (Week 3-4)

#### D1: CPG Gripper Wrapper
**Effort:** Medium (wrapper around existing env)
**Novel contribution:** First CPG-based gripper control for RL pick-and-place

New file: `cpg_wrapper.py`
```python
class CPGGripperWrapper(gym.Wrapper):
    """
    Converts raw gripper command to CPG-modulated smooth trajectory.
    Policy outputs: [dx, dy, dz, A (amplitude), ω (freq)]
    CPG outputs: smooth gripper signal = A * sin(ωt + learned_phase)
    """
    def __init__(self, env, cpg_dt=0.05):
        super().__init__(env)
        self.phi = 0.0
        self.cpg_dt = cpg_dt

        # Extended action space
        low = np.array([-0.05, -0.05, -0.05, 0.0, 0.5])
        high = np.array([ 0.05,  0.05,  0.05, 1.0, 4.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        dx, dy, dz = action[:3]
        A = np.clip(action[3], 0, 1)
        omega = np.clip(action[4], 0.5, 4.0)

        self.phi += omega * self.cpg_dt
        # Map CPG oscillation to gripper: [0=open, 1=close]
        gripper_cmd = A * np.sin(self.phi)  # oscillation in [-A, A]
        # Normalize to [-1, 1] for env
        gripper_norm = 2 * gripper_cmd - 1

        base_action = np.array([dx, dy, dz, gripper_norm], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(base_action)
        info['cpg_phase'] = self.phi % (2 * np.pi)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.phi = np.random.uniform(0, 2*np.pi)  # random phase at episode start
        return self.env.reset(**kwargs)
```

#### D2: Distributional SAC (QR-SAC) for Q-UCB Potential
**Effort:** Medium (modify critic network)
```python
# Quantile Regression SAC critic
class QuantileCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, n_quantiles=51):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_quantiles)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)  # (batch, n_quantiles)

    def q_ucb(self, obs, action, beta=1.5):
        quantiles = self.forward(obs, action)
        return quantiles.mean(dim=-1) + beta * quantiles.std(dim=-1)
```

#### D3: Self-Bootstrapped OT Reward
**Effort:** Medium (new module)
```python
class SelfBootstrappedOTReward:
    def __init__(self, max_demos=50):
        self.demos = []  # list of successful trajectory arrays
        self.max_demos = max_demos
        self.scale = 0.05

    def add_successful_episode(self, trajectory):
        """Call this when agent completes task successfully."""
        # Store reduced state: (EE_pos, cube_pos, gripper) — 7D
        reduced = trajectory[:, [14, 15, 16, 17, 18, 19, 26]]
        self.demos.append(reduced)
        if len(self.demos) > self.max_demos:
            self.demos.pop(0)

    def episode_reward(self, trajectory):
        if len(self.demos) == 0:
            return 0.0

        reduced = trajectory[:, [14, 15, 16, 17, 18, 19, 26]]

        # Compute min Wasserstein distance (approximated via sorted 1D projections)
        min_dist = float('inf')
        for demo in self.demos:
            # Subsample to same length
            n = min(len(reduced), len(demo))
            r = reduced[:n]; d = demo[:n]
            # Sum of per-dimension Wasserstein distances (approximation)
            dist = sum(
                wasserstein_distance(np.sort(r[:, i]), np.sort(d[:, i]))
                for i in range(7)
            )
            min_dist = min(min_dist, dist)

        return -self.scale * min_dist
```

---

### Phase E: Experimental Validation Plan

Run experiments in this order to measure each contribution independently:

| Experiment | Modification | Hypothesis | Metric |
|---|---|---|---|
| **E0 (baseline)** | Exp3 curriculum (11% success) | Reference point | Success rate |
| **E1** | A1: HER + dense reward | Better than pure HER | SR > 15% |
| **E2** | A2: Curriculum stability fix | Reduce regression | SR > 20% stable |
| **E3** | E2 + B1 barrier reward | Faster phase transitions | SR > 25% |
| **E4** | E3 + B2 Granger reward | Dense grasp signal | SR > 30% |
| **E5** | E4 + C1 contact priority | Better sample use | SR > 35% |
| **E6** | E5 + D1 CPG gripper | Smoother grasps | SR > 40% |
| **E7** | Full system: E6 + D2 + D3 | Grand unified | SR > 50% |

Each experiment: 1M steps SAC, same hyperparameters, 5-seed average for reliability.

---

### Phase F: Stretch Goals (Week 4+)

#### F1: Active Inference Agent
Replace reward-based RL with free energy minimization. Use `pymdp` library.
**Expected:** Elegant solution without reward engineering, but harder to implement.

#### F2: LLM-Adaptive Reward (EUREKA for our task)
Use Claude API to read diagnostic CSV data and iteratively refine the reward function code.
```python
def eureka_reward_iteration(csv_path, current_reward_code):
    """
    1. Read training diagnostics
    2. Prompt Claude with diagnostics + current reward code
    3. Claude generates improved reward code
    4. Run 100k steps with new reward
    5. Compare, repeat
    """
```

#### F3: Diffusion Policy from Successful Episodes
Use `diffusers` library to train a diffusion policy from Exp3's successful trajectories, then fine-tune online with OT reward.

---

## Part 6: Key Theoretical Justifications

### Why Granger Causality for Manipulation?
The fundamental problem with all current reward functions is that **proximity is not causality**. Being near the cube rewards the agent for reaching, not for grasping. Granger causality measures the *causal effect* of gripper commands on cube movement — which is exactly what grasping is. A perfectly stationary gripper near a cube has zero causal reward; a gripper actively squeezing the cube has maximum causal reward.

### Why Barrier Functions as Phase Gates?
The current one-time bonuses (+5, +15, +20) create discrete jumps in the reward landscape that are hard for the policy to exploit. Barrier functions create *continuous gradients* toward correct phase sequencing. The mathematical guarantee (PBRS theorem, Ng 1999): any potential-based shaping F(s) = Φ(s) preserves the optimal policy.

### Why CPG for Gripper?
The reach→grasp problem is fundamentally a temporal coordination problem: the gripper must close *at the right moment* with *the right force* and *the right speed*. Standard RL with step-wise actions treats each gripper command as independent, but grasping requires coordinated temporal sequences. A CPG is a natural inductive bias for temporally coordinated motor behavior — it's how biological systems solve this problem.

### Why Q-UCB Potential Shaping?
Our task has sparse terminal reward (+100). Most episodes provide zero signal toward task completion. The Q-function's *uncertainty* (upper confidence bound) is high near states never visited — which are exactly the states near task completion. Using Q_UCB as a potential provides directional gradient toward unexplored (potentially successful) states, solving the exploration problem without random noise.

### Why Optimal Transport from Self-Demonstrations?
OT reward using our own 11% success episodes is a form of **goal-conditioned imitation learning** that doesn't require human demonstrations. The agent learns to be more like its own best past self. As it improves, the demonstrations improve. This creates a self-supervised learning loop that doesn't exist in any current approach for this specific task.

---

## Part 7: Risk Assessment

| Idea | Risk | Mitigation |
|---|---|---|
| Granger reward | Causal model may overfit, wrong correlations | Small model, slow update rate, ablate |
| CPG wrapper | Oscillation may destabilize learning | Start with fixed ω, only learn A |
| Barrier functions | Wrong barrier design → reward hacking | Test each barrier independently |
| Q-UCB potential | Distributional SAC harder to tune | Use ensemble SAC as simpler alternative |
| OT reward | Needs ≥1 successful episode to start | Bootstrap from Exp3 saved episodes |
| Curriculum stability | May still regress | Add rollback mechanism |

---

## Part 8: Literature Foundation

Key papers referenced in this plan:

| Paper | Source | Relevance |
|---|---|---|
| Contact Energy Based Hindsight Experience Prioritization | TUM 2024 (arxiv:2312.02677) | CEBP replay priority |
| SLOPE: Shaping Landscapes with Optimistic Potential | 2026 (arxiv:2602.03201) | Q-UCB potential shaping |
| Barrier Functions Inspired Reward Shaping | IISc 2024 (arxiv:2403.01410) | Barrier function reward |
| Granger-Causal Hierarchical Skill Discovery (HIntS) | 2023 (arxiv:2306.09509) | Causal reward signal |
| Score-Based Diffusion with OT Reward (OTPR) | 2025 (arxiv:2502.12631) | OT-based reward |
| Bio-inspired CPG + RL | Nature 2025 | CPG gripper concept |
| Diffusion Meets Options | 2024 (arxiv:2410.02389) | Hierarchical diffusion skills |
| On-Manifold Exploration (SOE) | 2025 (arxiv:2509.19292) | Valid action manifold |
| Active Inference / Free Energy | Friston 2023 | Reward-free agent |
| CRISP: Curriculum-Inducing Primitive Subgoal Planning | 2023 (arxiv:2304.03535) | Curriculum stability |

---

## Summary: What to Do First

The highest-ROI actions in priority order:

1. **TODAY:** Enable HER on top of Exp2's dense reward (A1) — 1 line of code change, likely 10-20% success rate
2. **DAY 2:** Add barrier function reward to phase-gate the task (B1) — 10 lines in `_compute_reward`, expected 1.5× faster convergence
3. **DAY 3:** Fix curriculum advancement threshold (A2) — stop regression at Stage 3
4. **WEEK 2:** Implement Granger-causal reward (B2) — novel contribution, expected biggest improvement
5. **WEEK 2:** Implement contact-priority HER with temporal decay (C1) — novel contribution
6. **WEEK 3:** CPG gripper wrapper (D1) — novel architecture, cleaner grasping
7. **WEEK 4:** Distributional SAC + Q-UCB potential (D2) + Self-bootstrapped OT reward (D3) — grand unified system
