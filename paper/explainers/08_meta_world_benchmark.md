# 08 -- Meta-World Benchmark

## What is Meta-World?

Meta-World (Yu et al., 2020) is a benchmark for robotic manipulation reinforcement learning. It provides 50 distinct manipulation tasks in a simulated environment, all using the same robot: a **Sawyer robotic arm**.

It's one of the most widely used benchmarks for testing RL algorithms on robotics tasks, and it's the primary (and only) benchmark in our paper.

## The Sawyer Robot

The Sawyer is a 7-DOF (degrees of freedom) industrial robot arm made by Rethink Robotics. In Meta-World:
- It has a parallel-jaw gripper (two fingers that open and close)
- It operates above a table with objects
- The control interface is 4-dimensional: 3D end-effector position + gripper open/close

## Observations and Actions

### State observations (39 dimensions)
The agent receives a vector containing:
- Robot gripper position (x, y, z)
- Gripper state (open/closed amount)
- Object positions and orientations
- Goal position
- Various velocities

This is **state-based** observation -- the agent gets ground-truth positions, not camera images. Visual RL (using pixels) is a separate and harder problem that we acknowledge in our limitations.

### Actions (4 dimensions)
Each action is a 4D continuous vector:
- Delta x, y, z: how much to move the gripper
- Gripper: -1 (close) to +1 (open)

These values are clipped to [-1, 1] after the policy's tanh output.

### Episode length
Each episode runs for up to 500 steps. After 500 steps, the episode ends regardless of whether the task was completed.

## The Five Tasks in Our Paper

We selected tasks spanning a range of difficulties:

### Easy: drawer-close-v3
- Push a drawer closed
- Difficulty: Easy -- the drawer is right there, just push it
- All 8/8 seeds solve it
- Why it's easy: simple contact dynamics, large margin for error

### Medium: window-open-v3
- Slide a window open by grasping and pushing the handle
- Difficulty: Medium -- requires locating the handle
- All 8/8 seeds solve it

### Medium: door-open-v3
- Open a door by grasping the handle and pulling
- Difficulty: Medium -- requires understanding articulated motion
- All 8/8 seeds solve it

### Hard: peg-insert-side-v3
- Insert a peg sideways into a hole
- Difficulty: Hard -- requires precise alignment in 3D
- 11/12 seeds solve it (1 collapses)
- Why it's hard: the insertion requires precision; small errors mean the peg doesn't enter the hole

### Hard: pick-place-v3
- Pick up an object and place it at a goal location
- Difficulty: Hard -- requires a precise sequence: approach, grasp, lift, move, place
- 9/12 seeds ever solve it (but 3 of those forget the solution)
- Why it's hardest: it requires the most complex multi-step behavior. Each step must succeed for the task to succeed. Grasping alone is hard -- the gripper must be at exactly the right position and close at exactly the right moment.

## Reward Structure

This is critical for understanding our paper's findings.

Meta-World uses **dense, shaped rewards**:
- The reward increases as the agent gets closer to completing the task
- Each sub-goal (reaching the object, grasping it, moving it to the target) contributes to the reward
- Episode reward >= 500 means the task is considered solved

### Why the reward structure matters for alpha-bifurcation

The bifurcation phenomenon requires a reward landscape where:
1. Seeds that discover the task get a strong, clear signal (reward increases dramatically)
2. Seeds that fail to discover the task get a weak, uninformative signal (reward stays near zero)

This creates the asymmetry that drives the feedback loop:
- Successful seeds: reward -> informative Q-values -> policy improves -> entropy stays near target -> alpha moderate
- Failed seeds: no reward -> uninformative Q-values -> policy stagnates -> entropy drifts -> alpha collapses/explodes

### Why ManiSkill3 showed no bifurcation

We tried to replicate the finding on ManiSkill3 (another manipulation benchmark) but failed. ManiSkill3's rewards are uniformly shaped -- ALL seeds get rewards of 2-7 regardless of actual task progress. There's no clear separation between "found the solution" and "didn't find it" in the reward signal. So the feedback loop that drives bifurcation never forms.

This confirms that alpha-bifurcation is a property of the **reward landscape**, not just the SAC algorithm.

## Meta-World Versions

Meta-World has gone through several versions (v1, v2, v3), and McLean et al. (2025) documented undocumented behavioral changes between versions. We use **v3 throughout** for consistency. The version matters because task dynamics, reward functions, and even what constitutes "success" can differ.

## Why Meta-World (and not other benchmarks)?

1. **Standard benchmark**: Most cited manipulation RL benchmark
2. **Task diversity**: 50 tasks spanning various manipulation skills
3. **Dense rewards**: Provides the reward structure needed to observe bifurcation
4. **Difficulty spectrum**: Tasks range from trivially easy to quite hard, letting us show the bifurcation depends on difficulty
5. **Community trust**: Results on Meta-World are widely understood and comparable

---

**Next:** [09 -- Known Failure Modes in Deep RL](09_known_failure_modes_deep_rl.md)
