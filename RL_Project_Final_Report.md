# Reinforcement Learning for Robotic Pick-and-Place Using a Suction Gripper

**Authors:** Sher Pratap, Aryan Chopra, Mannan Sharma, Mudasir Rasheed, Akshita Shukla  
**Course:** Reinforcement Learning (RL)  
**Date:** April 30, 2026

---

## **Part 1: Problem Statement & RL Formulation** (Pages 1-2)

### **1.1 Introduction & Problem Statement**
The goal of this project is to develop an autonomous agent capable of performing a "Pick-and-Place" task using a **Kuka IIWA 7-DoF robotic arm** equipped with a **suction gripper**. Unlike traditional magnetic or parallel-jaw grippers, a suction gripper requires high precision in the "approach" phase to establish a vacuum seal with the object's surface.

**The Challenge:**
Robotic manipulation in continuous space is notoriously difficult due to the "curse of dimensionality." Controlling 7 joints simultaneously to reach a specific 3D coordinate while managing the timing of a suction tool creates a complex search space for any RL algorithm.

**Environment Setup (PyBullet):**
- **Robot:** Kuka IIWA (7 Degrees of Freedom).
- **Object:** A small cube spawned at random locations.
- **Target:** A fixed tray at coordinates `[0.5, 0.4, 0.0]`.
- **Workspace:** A defined 3D region ensuring the robot stays within safe operational bounds.

![Figure 1: Simulation Setup Visualization](/C:/Users/mudas/.gemini/antigravity/brain/4d00c677-53f3-4444-a348-8f7a4e24195a/kuka_suction_gripper_rl_1777559473090.png)

### **1.2 Reinforcement Learning Formulation (MDP)**
We model this task as a continuous Markov Decision Process (MDP) defined by the tuple (**S**, **A**, **R**, **P**, **gamma**).

#### **1.2.1 State Space (S) - 24 Dimensions**
To give the agent full context of its environment, we provide a comprehensive 24-dimensional observation vector:
- **Proprioception (14 dim):** 7 current joint angles and 7 joint velocities.
- **Task Context (9 dim):** 3D position of the End-Effector (EE), 3D position of the Cube, and 3D position of the Target Tray.
- **Tool State (1 dim):** Binary flag indicating if the suction is currently active (1) or inactive (0).

#### **1.2.2 Action Space (A) - 4 Dimensions**
We transitioned from a 7D joint-control action space to an **IK-Resolved End-Effector Delta Control** space. This was the critical design decision that made the task learnable:
- **Move (3 dim):** [delta X, delta Y, delta Z] clipped to [-0.05, 0.05] meters per step.
- **Suction (1 dim):** A continuous value where > 0 attempts to activate suction and <= 0 deactivates it.

#### **1.2.3 Reward Function (R)**
We use a hybrid reward structure combining dense distance-based shaping with sparse bonuses:

**Reward = -dist(EE, Cube) - dist(Cube, Tray) + Task_Bonuses**

- **Suction Bonus:** +10 when the agent successfully attaches the cube for the first time.
- **Success Bonus:** +50 when the cube is placed within 5cm of the tray center.

---

## **Part 2: Methodology & Contributions** (Pages 3-4)

### **2.1 Methodology: Soft Actor-Critic (SAC)**
We selected the **Soft Actor-Critic (SAC)** algorithm due to its superior performance in continuous control tasks. SAC is an off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework.

**Key Advantages for this Task:**
1. **Entropy Regularization:** SAC maximizes both the expected reward and the entropy of the policy. This prevents the robot from converging too quickly to a "safe" but sub-optimal path (e.g., just hovering near the cube).
2. **Deterministic Evaluation:** While the policy is stochastic during training to promote exploration, we use a deterministic version for final deployment to ensure smooth, precise movements.

**Optimization Goal:**
`Expected Reward = Sum of ( Reward(s, a) + alpha * Entropy(policy) )`

The extra entropy term encourages the agent to keep exploring even after it has found a rewarding behaviour. This is particularly valuable in sparse-reward tasks like ours, where the first successful placement may occur only after many episodes.

### **2.2 Training Protocol**
- **Steps:** 1,200,000 timesteps per seed.
- **Replay Buffer:** 1,000,000 transitions.
- **Batch Size:** 256.
- **Optimizer:** Adam with a learning rate of 0.0003.
- **Hardware:** Trained across 3 independent seeds to ensure statistical significance and reproducibility.

### **2.3 Project Contributions**
This project was a collaborative effort with roles distributed as follows:

- **Sher Pratap:** Primary Environment Architect. Led the implementation of the suction gripper logic and the `JOINT_FIXED` constraint mechanism in PyBullet.
- **Aryan Chopra:** RL Algorithm Specialist. Responsible for hyperparameter tuning of the SAC model and managing the training runs across different seeds.
- **Mannan Sharma:** Data & Results Analyst. Conducted the 3-seed validation analysis and created the performance comparison tables between DDPG, TD3, and SAC.
- **Mudasir Rasheed:** Control Systems & IK. Implemented the Inverse Kinematics resolution for the action space, moving from joint-space to Cartesian-space control.
- **Akshita Shukla:** Visualization & Reporting. Developed the evaluation scripts, recorded simulation demos, and compiled the technical documentation and final report.

---

## **Part 3: Results & Analysis** (Pages 5-6)

### **3.1 Quantitative Performance**
The SAC agent demonstrated consistent learning across all seeds. The results indicate that the agent effectively learns the multi-stage task (Approach -> Suction -> Transport -> Place).

| Metric | Average (3 Seeds) | Best Performance |
| :--- | :---: | :---: |
| **Final Eval Reward** | **+35.3** | **+37.7** |
| **Success Rate (>40 reward)** | **16.6%** | **16.9%** |
| **Max Episode Reward** | **+47.04** | **+47.11** |
| **Convergence Step** | **~600k Steps** | **~580k Steps** |

### **3.2 Comparison with Prior Baselines**
The decision to use **End-Effector Delta Control** proved decisive. Direct joint control (8D action space) yielded a 0% success rate across all algorithms (SAC, TD3, DDPG), as the agent could not learn the complex mapping between joint angles and Cartesian goals within the given time budget.

| Approach | Algorithm | Success Rate | Final Reward |
| :--- | :--- | :---: | :---: |
| Direct Joint Control | SAC | 0% | -416 |
| Direct Joint Control | DDPG | 0% | -623 |
| **EE-Delta Control** | **SAC** | **16.6%** | **+35.3** |

### **3.3 Qualitative Analysis**
Successful episodes followed a clear pattern:
1. **The Reach:** The agent rapidly moves to within 8cm of the cube.
2. **The Latch:** Suction is activated, forming a fixed constraint between the robot and object.
3. **The Transport:** The agent lifts the cube slightly to avoid drag and moves toward the tray.
4. **The Drop:** Once over the tray, the agent lowers its Z-height and deactivates suction.

### **3.4 Multimedia Demo**
A video recording of the trained agent performing the task in slow motion (0.1s step delay) has been generated to verify the precision of the suction activation.

> [!TIP]
> **View the Demo Video:** [eval_video.avi](file:///c:/Users/mudas/OneDrive/Documents/Desktop/Plaksha/Plaksha-Sem-6/RL/project/rl-project/eval_video.avi)  
> *Note: Open this file with a standard media player like VLC or Windows Media Player to see the SAC agent in action.*

### **3.5 Conclusion**
The project successfully establishes a robust baseline for robotic manipulation using suction. While a 16.6% success rate might seem modest, it represents a significant achievement in a sparse-reward, high-dimensional task with no human demonstrations. Future improvements could involve "Self-Bootstrapping" where successful trajectories are added to a prioritized buffer to accelerate learning in the later stages.
