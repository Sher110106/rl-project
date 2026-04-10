# 03 -- Q-Functions and the Overestimation Problem

## What is a Q-Function?

The Q-function Q(s, a) answers: **"If I'm in state s, take action a, and then follow my policy from there, what total return will I get?"**

It's a prediction of cumulative future reward. A good Q-function tells the agent which actions are worth taking in which states.

In deep RL, the Q-function is approximated by a neural network (called the **critic**). The network takes (state, action) as input and outputs a single number -- the estimated Q-value.

## How Q-Functions Learn (Temporal Difference)

The Q-function learns using **Bellman's equation**, which says:

```
Q(s, a) = r + gamma * Q(s', a')
```

"The value of being in state s and doing a equals the immediate reward r, plus the discounted value of whatever comes next."

We turn this into a loss function:

```
Loss = (Q(s, a) - [r + gamma * Q_target(s', a')])^2
```

where Q_target is a slowly-updated copy of Q (the "target network"), used for stability.

## The Overestimation Problem

Here's the problem: Q-functions tend to **overestimate** values. Why?

When we compute the target, we pick the best action at the next state:

```
target = r + gamma * max_a' Q(s', a')
```

The max operator creates a systematic positive bias. If the Q-function has any noise (it always does -- it's a neural network), taking the max over noisy estimates will tend to pick the estimate that's most inflated by noise. It's like asking "what's the temperature?" to 10 people, some of whom guess too high and some too low, then always believing the highest answer.

Over many updates, this bias compounds. Q-values drift higher and higher, detached from reality. This is called **Q-value overestimation** or **Q-value divergence**.

### Why it matters for our paper

In our ablation study, when we replace auto-entropy tuning with fixed annealing, the policy becomes near-deterministic. This makes overestimation *worse* because:
- A near-deterministic policy only visits a narrow slice of the state-action space
- The critic overfits to that narrow region
- Q-values spike to 12,000-17,000 (completely unrealistic)
- Then collapse when the policy shifts even slightly

This is one of the mechanistic signatures we observe in the paper (Section 5.5, Figure 4).

## Twin Q-Networks (the TD3 trick)

**TD3** (Twin Delayed DDPG, Fujimoto et al., 2018) proposed a simple fix: use **two** independent Q-networks, Q1 and Q2, and take the minimum:

```
target = r + gamma * min(Q1(s', a'), Q2(s', a'))
```

By always taking the pessimistic estimate, you counteract the optimistic bias of the max. It doesn't eliminate overestimation entirely, but it significantly reduces it.

**SAC uses this same trick.** It maintains two Q-networks and takes the minimum for computing targets. When the paper says "Q-probe analysis" and reports Q-values, it's looking at min(Q1, Q2) at fixed probe states.

## Target Networks and Soft Updates

To prevent the Q-function from chasing a moving target (the target depends on the Q-function itself), we use a **target network** -- a separate copy of Q that updates slowly.

Instead of copying Q to Q_target every N steps (hard update), SAC uses **soft updates** (also called Polyak averaging):

```
Q_target = tau * Q + (1 - tau) * Q_target
```

With tau = 0.005 (our paper), the target network slowly blends toward the current Q-network. This creates a smoother, more stable learning signal.

## Q-Value Probes (What We Do in the Paper)

To monitor what the Q-function is doing across training, we sample a fixed set of **probe states** early in training and track their Q-values over time. This gives us a diagnostic window into the critic's health:

- **Stable, moderate Q-values**: Healthy learning. The critic's estimates are reasonable and consistent.
- **Q-values spiking to extreme values then crashing**: Overestimation followed by collapse. This happens when the policy becomes too deterministic (low entropy) -- exactly what we see under fixed annealing.

The probe analysis is what provides the mechanistic evidence linking low alpha (deterministic policy) to Q-value pathology.

---

**Next:** [04 -- Replay Buffers and Off-Policy Learning](04_replay_buffer_and_off_policy.md)
