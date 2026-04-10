# Paper Explainers -- Reading Roadmap

**Paper:** *SAC's Entropy Coefficient as an Implicit Success Signal in Robotic Manipulation*
**Venue:** CoRL 2026

This set of documents explains every concept involved in the paper, from the ground up. After reading them in order, you should be able to explain the full paper -- what each piece is, why it matters, and how they all connect.

---

## Learning Path

### Part I: Foundations (What you need to know before the paper)

| # | File | What it covers |
|---|------|----------------|
| 01 | [Reinforcement Learning Basics](01_reinforcement_learning_basics.md) | States, actions, rewards, policies, episodes, returns, discount factor |
| 02 | [Policy Gradients and Actor-Critic](02_policy_gradient_and_actor_critic.md) | How agents learn policies, the actor-critic architecture |
| 03 | [Q-Functions and the Overestimation Problem](03_q_functions_and_overestimation.md) | Q-learning, twin critics (TD3), why Q-values blow up |
| 04 | [Replay Buffers and Off-Policy Learning](04_replay_buffer_and_off_policy.md) | Experience replay, on-policy vs off-policy, sample efficiency |
| 05 | [Entropy in Reinforcement Learning](05_entropy_in_rl.md) | Information-theoretic entropy, maximum entropy RL framework |
| 06 | [Soft Actor-Critic (SAC)](06_soft_actor_critic.md) | The full SAC algorithm -- architecture, objective, why it works |
| 07 | [Auto-Entropy Tuning (the alpha mechanism)](07_auto_entropy_tuning.md) | How alpha is learned, target entropy, the dual optimization, the feedback loop |

### Part II: Experimental Context

| # | File | What it covers |
|---|------|----------------|
| 08 | [Meta-World Benchmark](08_meta_world_benchmark.md) | The Sawyer robot, the 50 tasks, reward structure, why it's used |
| 09 | [Known Failure Modes in Deep RL](09_known_failure_modes_deep_rl.md) | Primacy bias, plasticity loss, catastrophic forgetting -- prior work |
| 10 | [Ablation Studies and Experimental Design](10_ablation_studies.md) | What ablations are, factorial design, why they matter for causal claims |

### Part III: Our Contributions (What the paper actually shows)

| # | File | What it covers |
|---|------|----------------|
| 11 | [The Alpha-Bifurcation Phenomenon](11_alpha_bifurcation.md) | The core discovery: alpha trajectories split and predict success/failure |
| 12 | [Three Failure Modes](12_three_failure_modes.md) | Alpha-collapse, alpha-explosion, discover-then-forget |
| 13 | [The Ablation: Fixed Annealing Destroys SAC](13_ablation_results.md) | The 2x2 experiment proving auto-tuning is load-bearing |
| 14 | [Alpha as a Training Diagnostic](14_alpha_as_diagnostic.md) | Practical monitoring, thresholds, early stopping, what alpha can and can't tell you |

---

## Quick Summary for the Impatient

SAC is the standard algorithm for robot manipulation RL. It has a parameter called alpha that automatically adjusts exploration vs exploitation. Everyone treats alpha as a boring hyperparameter. We found that alpha actually *predicts* whether training will succeed or fail -- hundreds of thousands of steps before you'd know from the reward. When you replace auto-tuning with a fixed schedule, performance drops by up to 98%. Alpha isn't just a tuning knob -- it's an implicit training diagnostic that people have been ignoring.
