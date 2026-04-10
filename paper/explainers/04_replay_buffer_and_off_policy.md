# 04 -- Replay Buffers and Off-Policy Learning

## On-Policy vs Off-Policy

### On-policy (e.g., PPO, A2C)
The agent can only learn from data collected by its **current** policy. Once you update the policy, all the old data is thrown away. This is like a student who can only learn from their own most recent exam -- wasteful.

### Off-policy (e.g., SAC, TD3, DQN)
The agent can learn from data collected by **any** policy, including past versions of itself. This means you can store and reuse past experience. This is far more sample-efficient.

**SAC is off-policy.** This is one of the main reasons it's the standard for robotics -- sample efficiency matters a lot when each interaction with a real robot (or even a simulator) is expensive.

## What is a Replay Buffer?

A replay buffer (also called experience replay) is a database that stores past interactions:

```
buffer = [(s1, a1, r1, s1', done1),
          (s2, a2, r2, s2', done2),
          ... up to N entries]
```

Each entry is a **transition**: the state, the action taken, the reward received, the next state, and whether the episode ended.

In our paper, the replay buffer holds up to **1,000,000 transitions**. When it's full, the oldest transitions get overwritten (FIFO).

## How it's Used in Training

Each training step:
1. Agent collects new experience by interacting with the environment
2. New transitions are added to the buffer
3. A random mini-batch of 256 transitions is sampled from the buffer
4. The critic (Q-networks) and actor (policy) are updated using this batch

The random sampling is key -- it breaks the temporal correlation between consecutive transitions, which would otherwise cause the neural network to overfit to recent patterns.

## Why Replay Buffers Matter for Our Paper

### 1. Buffer Success Fraction as a Diagnostic
In our experiments (Section 5.5, Figure 4), we track the **fraction of successful transitions** in the replay buffer over time. This tells us:
- How quickly the agent is discovering successful behaviors
- Whether successful experience is accumulating or stagnating

We find that auto-entropy (Method A) accumulates successful experience ~2x faster than annealed methods. The near-deterministic policy under annealing explores less, so it finds fewer successful trajectories to learn from.

### 2. The Demo Reward's k-NN Buffer
In the ablation (Methods C and D), we test a "self-bootstrapped demo reward" that uses a k-nearest-neighbors approach on the replay buffer:
- It identifies the k=5 most similar successful transitions in the buffer
- It gives a bonus reward based on distance to those transitions
- The idea: reward the agent for being in states similar to past successes

This is essentially using the replay buffer as a memory of "what worked" to shape the reward. In practice, it didn't help -- it actually introduced noise into the Q-function's learning.

### 3. Buffer Composition Reflects Training Dynamics
The replay buffer is a physical record of the agent's exploration history:
- A buffer full of diverse, successful transitions -> healthy training
- A buffer full of repetitive, failed transitions -> the agent is stuck
- A buffer where success fraction plateaus -> exploration has stalled

This connects directly to alpha: when alpha collapses (exploration stops), the buffer stops accumulating diverse new experience, and learning stalls.

## Buffer Size Tradeoffs

- **Too small**: The agent forgets past experience too quickly. Important successful trajectories may be overwritten before they're learned from.
- **Too large**: The agent spends too much time learning from outdated experience (collected by a much earlier, worse policy). This can slow down learning.
- **1M (our choice)**: Standard for Meta-World. Large enough to retain important early discoveries, small enough that the data stays relevant.

---

**Next:** [05 -- Entropy in Reinforcement Learning](05_entropy_in_rl.md)
