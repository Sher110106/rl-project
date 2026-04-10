# 06 -- Soft Actor-Critic (SAC)

## What is SAC?

**Soft Actor-Critic** (Haarnoja et al., 2018) is the most widely used RL algorithm for continuous-control robotics. It combines:
- Actor-critic architecture (a policy network + value networks)
- Maximum entropy framework (reward + entropy bonus)
- Off-policy learning (replay buffer for sample efficiency)
- Twin Q-networks (to combat overestimation)

It is called "soft" because the policy is not a hard argmax over Q-values (like in DQN), but a soft, stochastic distribution that balances reward-seeking with entropy.

## SAC's Objective

SAC maximizes the **maximum entropy objective**:

```
J(pi) = sum_t E[ r(s_t, a_t) + alpha * H(pi(.|s_t)) ]
```

In words: maximize the sum of rewards PLUS alpha times the entropy of the policy at each step.

This is equivalent to:
```
J(pi) = sum_t E[ r(s_t, a_t) - alpha * log pi(a_t|s_t) ]
```

Since H(pi) = -E[log pi(a|s)], the entropy bonus is just subtracting alpha * log probability of the action. Actions that are "surprising" (low probability) get a bonus; obvious actions get penalized.

## SAC's Architecture

SAC uses **five neural networks** (three if you count shared parameters):

### 1. Actor (Policy Network)
- Input: state s (39 dimensions in Meta-World)
- Output: mean mu and log-std for each action dimension
- Architecture: MLP with 2 hidden layers of 256 units each
- Samples actions from: a = tanh(mu + sigma * noise), where noise ~ N(0, 1)

### 2. Critic 1 (Q-network 1)
- Input: (state, action) concatenated
- Output: single scalar Q-value
- Architecture: MLP with 2 hidden layers of 256 units each

### 3. Critic 2 (Q-network 2)
- Same architecture as Critic 1 but independently initialized
- Used for the twin Q-trick (take the min to reduce overestimation)

### 4. Target Critic 1 (slowly updated copy of Critic 1)
### 5. Target Critic 2 (slowly updated copy of Critic 2)

Target networks are updated via soft/Polyak updates:
```
Q_target = tau * Q + (1 - tau) * Q_target    (tau = 0.005)
```

## SAC's Training Loop

Each training step:

### Step 1: Collect experience
```
a ~ pi(.|s)        # Sample action from current policy
s', r, done = env.step(a)   # Execute in environment
buffer.add(s, a, r, s', done)  # Store in replay buffer
```

### Step 2: Sample a batch
```
batch = buffer.sample(256)   # Random mini-batch of 256 transitions
```

### Step 3: Update Critics
For each transition (s, a, r, s', done) in the batch:
```
a' ~ pi(.|s')   # Sample next action from current policy
Q_target = r + gamma * (min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a'|s'))
```

Note the entropy term: the target includes `-alpha * log pi(a'|s')`, meaning high-entropy future actions have higher value.

```
Loss_Q1 = mean((Q1(s, a) - Q_target)^2)
Loss_Q2 = mean((Q2(s, a) - Q_target)^2)
```

### Step 4: Update Actor
The actor is updated to maximize:
```
J_pi = E[min(Q1(s, a_new), Q2(s, a_new)) - alpha * log pi(a_new|s)]
```
where a_new is freshly sampled from the current policy.

In words: choose actions that the critic thinks are good (high Q), while also maintaining high entropy (low log probability).

### Step 5: Update Alpha (if auto-tuning)
See the next explainer (07) for details.

### Step 6: Soft-update target networks
```
Q1_target = 0.005 * Q1 + 0.995 * Q1_target
Q2_target = 0.005 * Q2 + 0.995 * Q2_target
```

## Why SAC for Robotics?

### 1. Sample efficiency (off-policy)
SAC reuses past experience via the replay buffer. This is critical for robotics where data is expensive to collect.

### 2. Stability
The twin Q-networks and soft updates prevent the catastrophic instability seen in earlier algorithms (like DDPG).

### 3. Exploration (entropy)
The entropy bonus provides principled exploration without needing to manually tune noise parameters. The agent naturally explores in early training and focuses in later training.

### 4. Continuous actions
SAC handles continuous action spaces natively via its squashed Gaussian policy.

### 5. Hyperparameter robustness
SAC with auto-entropy tuning works well across a range of tasks without much hyperparameter tuning -- you basically set the learning rate, network size, and buffer size, and it works. This is why it's the default in libraries like Stable-Baselines3.

## SAC vs Other Algorithms

| Feature | DQN | DDPG | TD3 | PPO | SAC |
|---------|-----|------|-----|-----|-----|
| Action space | Discrete | Continuous | Continuous | Both | Continuous |
| On/Off-policy | Off | Off | Off | On | Off |
| Stochastic policy | No | No | No | Yes | Yes |
| Entropy regularization | No | No | No | Partial | Yes |
| Twin Q-networks | No | No | Yes | No | Yes |
| Sample efficiency | High | Moderate | Moderate | Low | High |

SAC is essentially "TD3 but with a stochastic policy and entropy regularization," which turns out to be a very powerful combination.

## The Key Thing for Our Paper

SAC's entropy regularization isn't just a nice-to-have -- it's the core mechanism that enables learning on hard manipulation tasks. The alpha parameter that controls it isn't just a hyperparameter -- its trajectory over training encodes whether the agent is learning, stuck, or diverging. Replacing the auto-tuning with a fixed schedule breaks this mechanism catastrophically.

---

**Next:** [07 -- Auto-Entropy Tuning (the Alpha Mechanism)](07_auto_entropy_tuning.md)
