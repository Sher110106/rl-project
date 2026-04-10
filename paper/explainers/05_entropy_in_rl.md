# 05 -- Entropy in Reinforcement Learning

## What is Entropy?

In information theory, **entropy** measures the randomness or uncertainty of a probability distribution.

```
H(pi) = -E[log pi(a|s)] = -sum of pi(a|s) * log(pi(a|s)) for all a
```

Intuition:
- **High entropy**: The distribution is spread out. Many actions are roughly equally likely. The agent is **exploring** -- trying many different things.
- **Low entropy**: The distribution is concentrated. One action is much more likely than others. The agent is **exploiting** -- committing to what it thinks is best.
- **Zero entropy**: The policy is deterministic. One action has probability 1, all others have probability 0.

### Example
Imagine the agent choosing between 4 directions:
- Uniform: [0.25, 0.25, 0.25, 0.25] -> entropy = 1.39 (maximum randomness)
- Slightly biased: [0.4, 0.3, 0.2, 0.1] -> entropy = 1.28
- Very confident: [0.9, 0.05, 0.03, 0.02] -> entropy = 0.47
- Deterministic: [1.0, 0, 0, 0] -> entropy = 0

For SAC with continuous actions (Gaussian distribution), entropy depends on the standard deviation of the Gaussian. Wider Gaussian = higher entropy = more exploration.

## Why Entropy Matters in RL

The fundamental tension in RL is **exploration vs exploitation**:
- **Explore**: Try new things to discover better strategies
- **Exploit**: Use what you already know works to get high rewards

Without enough exploration, the agent gets stuck in **local optima** -- solutions that are OK but not great. It never discovers that there's something much better because it never looks.

Without enough exploitation, the agent wastes time trying random things and never converges on a good solution.

Entropy is the knob that controls this tradeoff.

## Maximum Entropy Reinforcement Learning

Standard RL objective:
```
Maximize: E[sum of rewards]
```

Maximum entropy RL objective (Ziebart, 2010):
```
Maximize: E[sum of (rewards + alpha * entropy)]
```

The agent now tries to maximize reward **while also keeping its policy as random as possible**. The extra entropy term says: "Among all policies that achieve high reward, prefer the one that's most random."

Why would you want randomness?

### 1. Better exploration
High-entropy policies try diverse actions, making them more likely to discover successful strategies. This is especially important in robotic manipulation where the agent needs to discover specific sequences (reach, grasp, lift, move, place).

### 2. Robustness
A policy that succeeds through many different action patterns is more robust than one that relies on a single precise sequence. If conditions change slightly (different object position, sensor noise), the diverse policy is more likely to still work.

### 3. Faster learning
By maintaining diversity in its behavior, the agent collects more informative training data, speeding up learning.

### 4. Multiple solutions
Many tasks have multiple valid solutions. Maximum entropy RL discovers all of them rather than collapsing onto one, which can be useful for transfer and adaptation.

## The Alpha Parameter

**Alpha (alpha)** is the **entropy coefficient** (also called the "temperature"). It controls the tradeoff:

```
Objective = rewards + alpha * entropy
```

- **Large alpha**: Entropy matters a lot. The agent prioritizes being random. More exploration, less exploitation.
- **Small alpha**: Entropy doesn't matter much. The agent prioritizes reward. More exploitation, less exploration.
- **Alpha = 0**: Standard RL. No entropy bonus. The agent only cares about reward.

The central question of our paper: **Should alpha be fixed, manually scheduled, or automatically learned?** And what happens in each case?

## Target Entropy

In SAC's auto-tuning formulation, you don't set alpha directly. Instead, you set a **target entropy** -- the level of randomness you want the policy to maintain.

The default target entropy in Stable-Baselines3 is:
```
H_target = -dim(action_space) = -4 for Meta-World
```

This is a heuristic: for a 4-dimensional action space, target entropy of -4 means "keep roughly 1 nat of entropy per action dimension." (The negative sign is because log probabilities are negative.)

The auto-tuning mechanism adjusts alpha up or down to keep the policy's actual entropy close to this target. This is detailed in the next explainer.

## Entropy and Our Paper's Core Finding

Our discovery is that the *dynamics* of entropy -- how it changes over training -- tell you whether training is succeeding or failing. On hard tasks:

- **Healthy training**: Entropy stays near the target. Alpha stays in a moderate range (0.02-0.25). The agent is actively exploring and learning.
- **Collapse**: Entropy drops way below target. Alpha collapses toward 0. The agent stopped exploring and is stuck.
- **Explosion**: Entropy stays way above target. Alpha grows unboundedly. The agent is too random and can't learn anything useful.

The key insight is that alpha -- the Lagrange multiplier maintaining the entropy target -- is a diagnostic signal recording which dynamic the agent has entered. No one was watching it. We argue they should be.

---

**Next:** [06 -- Soft Actor-Critic (SAC)](06_soft_actor_critic.md)
