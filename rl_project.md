# Reinforcement Learning for Robotic Pick-and-Place Using a Suction Gripper

**Authors:** Sher Pratap, Aryan Chopra, Mannan Sharma, Mudasir Rasheed, Akshita Shukla  
**Department:** [Your Department]  
**Institution:** [Your Institution]

---

## **Part 1: Problem Statement & RL Formulation (Pages 1-2)**

### **Abstract**
This paper presents a reinforcement learning (RL) framework for teaching a Kuka IIWA 7-DoF robotic arm to perform pick-and-place tasks using a suction gripper. The task is modeled as a Markov Decision Process (MDP) with continuous state and action spaces. We compare two action-space formulations: direct joint-angle control and end-effector delta control (Cartesian displacement + inverse kinematics). The Soft Actor-Critic (SAC) algorithm is trained for 1.2 million steps (~25,000 episodes) in a PyBullet simulation. Our results show that end-effector delta control enables the agent to achieve a 16.6% success rate, whereas direct joint control fails completely. Furthermore, SAC significantly outperforms both TD3 and DDPG on this sparse-reward manipulation task. We analyse the reasons for this performance gap and discuss the critical role of action-space design and entropy-regularised exploration.

### **I. Introduction**
Robotic manipulation is one of the most demanding challenges in intelligent systems. A seemingly simple task—picking an object up and placing it elsewhere—actually requires the robot to solve several interconnected problems: perceiving the environment, planning a collision-free path, coordinating the motion of seven joints, and delicately controlling a gripper. Traditional engineering methods rely on hand-crafted rules and precise geometric models, which become brittle when the object, goal, or surroundings change even slightly.

Reinforcement learning offers a different path: the robot learns a policy by trial and error, interacting with the environment and receiving rewards. This paper explores how an RL agent can be trained to pick up a cube with a suction gripper and place it onto a fixed tray, using only low-level feedback—the robot’s own joint positions and the positions of the cube and tray. No human demonstrations, no pre-programmed grasp poses, and no dense reward shaping are used.

### **II. Problem Formulation**

#### **A. Robotic Task Description**
The simulation environment is built in PyBullet. It contains:
- A Kuka IIWA 7-DoF robotic arm, mounted statically at world coordinate [0, 0, 0].
- A cube (the object to be manipulated), whose position is randomly selected at the beginning of each episode within the workspace.
- A tray (the target location), permanently centred at [0.5 m, 0.4 m, 0 m].
- A workspace shaped as a three-dimensional rectangular prism of dimensions: length = 1.1 m, breadth = 0.5 m, height = 0.7 m.

The arm is equipped with a suction gripper. When activated, it can attach to the cube if the distance between the gripper and the cube’s surface is less than 8 cm. The task is considered successful when the cube is placed within 5 cm of the tray centre.

#### **B. Markov Decision Process (MDP) Formulation**
The pick-and-place problem is formalised as an MDP defined by the tuple (S, A, R, P, gamma).

**1) State Space S:** The state is a 24-dimensional real-valued vector containing 7 joint angles, 7 joint velocities, three 3-dimensional position vectors (Cube, Tray, End-effector), and a suction activation flag.

**2) Action Space A:** We test two different action-space formulations:
- **Direct joint control:** 8D [7 target joints + 1 suction command].
- **End-effector delta control:** 4D [dx, dy, dz, suction_action].

**3) Reward Function R:** 
`Reward = -dist(EE, Cube) - dist(Cube, Tray) + Suction_Bonus(+10) + Placement_Bonus(+50)`

---

## **Part 2: Methodology & Contributions (Pages 3-4)**

### **III. Methodology**

#### **A. Soft Actor-Critic (SAC)**
SAC is an off-policy, maximum-entropy reinforcement learning algorithm. It maintains a stochastic policy (actor) and two Q-value networks (critics). The policy maximises the expected reward plus an entropy term, which encourages exploration. This is particularly valuable in sparse-reward tasks like ours, where the first successful placement may occur only after many episodes.

#### **B. Training Setup**
We trained three independent SAC agents with different random seeds.
- Learning rate: 0.0003
- Replay buffer size: 1,000,000
- Batch size: 256
- Steps: 1,200,000 (~25,000 episodes)
- Runtime: ~8.5 hours per seed

#### **C. Evaluation Protocol**
During training, we tested the agent every 10,000 time steps. Each test consisted of 5 episodes run with the deterministic policy. We recorded the final evaluation mean, peak evaluation mean, and success rate (proportion of episodes where the reward exceeded 35).

### **IV. Contributions**

- **Sher Pratap:** Primary Environment Architect. Led the implementation of the suction gripper logic and the joint constraint mechanism in PyBullet.
- **Aryan Chopra:** RL Algorithm Specialist. Responsible for hyperparameter tuning of the SAC model and managing the training runs across different seeds.
- **Mannan Sharma:** Data & Results Analyst. Conducted the 3-seed validation analysis and created the performance comparison tables between DDPG, TD3, and SAC.
- **Mudasir Rasheed:** Control Systems & IK. Implemented the Inverse Kinematics resolution for the action space, moving from joint-space to Cartesian-space control.
- **Akshita Shukla:** Reporting & Documentation. Compiled the technical data, analysed the training curves, and authored the final report structure.

---

## **Part 3: Results & Analysis (Pages 5-6)**

### **V. Results**

#### **A. Performance with End-Effector Delta Control (SAC)**
Table I summarises the results of the three SAC agents using end-effector delta control.

| Metric | Seed 0 | Seed 1 | Seed 2 | Average |
| :--- | :---: | :---: | :---: | :---: |
| Final eval mean | +37.7 | +34.2 | +34.1 | **+35.3** |
| Peak eval mean | +42.3 | +43.3 | +43.2 | **+42.9** |
| Success rate | 16.4% | 16.9% | 16.6% | **16.6%** |

#### **B. Comparison with Direct Joint Control**
None of the agents using direct joint control achieved a single successful placement.

| Approach | Action Dim | Final Eval Mean | Success Rate |
| :--- | :---: | :---: | :---: |
| Direct Joints (SAC) | 8 | -416 | 0% |
| Direct Joints (TD3) | 8 | -482 | 0% |
| Direct Joints (DDPG) | 8 | -623 | 0% |
| **EE-Delta (SAC)** | **4** | **+35.3** | **16.6%** |

### **VI. Conclusions**
We have shown that a SAC agent using end-effector delta control can successfully learn a pick-and-place task with a suction gripper. acting in Cartesian space and delegating inverse kinematics to the simulator simplifies the learning problem dramatically. Future work will involve training for longer steps (2-3 million) and experimenting with self-bootstrapping demonstration buffers.

### **Acknowledgment**
We thank the developers of PyBullet and Stable-Baselines3 for providing the tools that enabled this work.

### **References**
[1] T. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning," 2018.  
[2] A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," 2021.  
[3] E. Coumans, "PyBullet," http://pybullet.org.
