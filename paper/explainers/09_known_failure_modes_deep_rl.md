# 09 -- Known Failure Modes in Deep RL

Before understanding our paper's contribution, it helps to know what failure modes were already documented. Our paper identifies new ones (alpha-collapse, alpha-explosion) and connects to existing ones (primacy bias, plasticity loss).

## 1. Primacy Bias (Nikishin et al., 2022)

### What it is
The agent's value function overfits to early training data. Experiences collected in the first few thousand steps disproportionately shape what the network learns, and it struggles to update its beliefs even when later data contradicts them.

### Why it happens
Neural networks are prone to "anchoring" on early patterns. The Q-network forms strong initial estimates from random, uninformative experience. These estimates create a gravitational pull -- later updates struggle to overcome the initial fit because the loss landscape of the network has already been shaped around it.

### Connection to our work
Alpha-collapse may be an **early indicator** of primacy bias:
- The policy collapses to near-deterministic behavior early on
- This means the agent only collects experience from a narrow slice of the state space
- The Q-network overfits to this narrow region
- The agent lacks the diverse experience needed to overcome its early beliefs
- Alpha collapsing to zero removes the exploration pressure that would generate the diversity needed to escape

In other words: primacy bias describes what happens in the Q-network, and alpha-collapse describes the corresponding signal in the entropy coefficient.

## 2. Plasticity Loss (Lyle et al., 2023)

### What it is
Over the course of training, a neural network progressively loses its ability to learn new things. Even when presented with clearly useful data, the network's parameters have "hardened" into a configuration that can't represent new solutions.

### Why it happens
Several mechanisms:
- Dead neurons (neurons that always output 0 and stop participating in learning)
- Saturated neurons (neurons stuck at extreme activation values)
- Effective rank collapse (the network uses fewer and fewer of its dimensions)
- The network's representational capacity gets consumed by early learning

### Connection to our work
Plasticity loss is consistent with our **discover-then-forget (DTF)** failure mode:
- The agent discovers a solution (reward spikes)
- But then catastrophically loses it (reward drops back to near-zero)
- The alpha stays in the healthy range (the entropy mechanism is fine)
- The network has lost the *capacity* to retain the solution

This is why alpha alone can't detect DTF: it's not an entropy problem, it's a network capacity problem. You need to monitor both alpha (exploration health) and reward (solution retention).

## 3. Q-Value Overestimation (Fujimoto et al., 2018)

### What it is
The Q-function systematically overestimates the value of state-action pairs. Values drift higher than reality, and the policy makes bad decisions based on inflated estimates.

(See the full explanation in [03 -- Q-Functions](03_q_functions_and_overestimation.md))

### Connection to our work
In our ablation, methods with fixed annealing (B, D) produce Q-value spikes to 12,000-17,000 followed by collapse. The mechanism:

1. Annealing forces alpha to a low value
2. Low alpha -> near-deterministic policy
3. Near-deterministic policy visits the same states repeatedly
4. Q-network overfits to those states, driving Q-values artificially high
5. When the policy shifts even slightly, the inflated Q-values become inaccurate
6. Q-values crash, destabilizing learning

Auto-tuning (Method A) avoids this: moderate alpha -> diverse exploration -> Q-network sees varied data -> stable Q-values.

## 4. Catastrophic Forgetting

### What it is
A neural network trained on new data forgets what it previously learned. This is a general problem in neural networks, not specific to RL.

### Connection to our work
The discover-then-forget failure mode is a form of catastrophic forgetting in the RL context. The policy network overwrites its learned picking behavior with new, possibly incompatible, parameter updates. The replay buffer helps somewhat (it retains old successful transitions), but can't fully prevent the network from "unlearning."

## 5. Reward Hacking / Reward Misspecification

### What it is
The agent finds a way to get high reward without actually solving the intended task. For example, a robot might learn to hover near the goal location (getting partial reward for proximity) without ever actually placing the object.

### Connection to our work
Not directly observed in our experiments, but relevant context. Meta-World's shaped rewards are designed to avoid this, but the reward structure is critical -- it's what creates the signal separation that drives alpha-bifurcation.

## Why Our Failure Modes are New

Prior work documented failures in the value network (overestimation, plasticity loss) or in the training process (primacy bias). We add failures visible in the **entropy mechanism**:

| Failure mode | Where it's visible | Prior work |
|---|---|---|
| Primacy bias | Q-values, learning curves | Nikishin 2022 |
| Plasticity loss | Learning curves, network rank | Lyle 2023 |
| Q-value overestimation | Q-values | Fujimoto 2018 |
| **Alpha-collapse** | **Alpha trajectory** | **Ours** |
| **Alpha-explosion** | **Alpha trajectory** | **Ours** |
| **Discover-then-forget** | **Reward trajectory + alpha** | **Ours (DTF)** |

The value of identifying these in the alpha trajectory is that alpha is a single scalar that's already computed during training. You don't need to do invasive analysis of the Q-network or track effective rank -- just log one number every few thousand steps.

---

**Next:** [10 -- Ablation Studies and Experimental Design](10_ablation_studies.md)
