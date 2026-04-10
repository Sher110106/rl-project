# 01 -- Reinforcement Learning Basics

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. Unlike supervised learning (where you have labeled data), the agent learns through trial and error -- it takes actions, observes what happens, and gets rewards or penalties.

Think of it like teaching a dog a trick: you don't show it exactly what to do step by step. Instead, you let it try things and reward it when it does something right.

## The Core Components

### Agent
The decision-maker. In our paper, this is a neural network controlling a Sawyer robotic arm.

### Environment
The world the agent interacts with. In our paper, this is Meta-World -- a physics simulation of a robotic arm and objects on a table.

### State (s)
A description of the current situation. In Meta-World, the state is a 39-dimensional vector that includes:
- Position and velocity of the robot's gripper (end effector)
- Joint angles
- Position of the object being manipulated
- Position of the goal location

### Action (a)
What the agent decides to do. In Meta-World, actions are 4-dimensional continuous vectors:
- 3 dimensions for moving the gripper (x, y, z)
- 1 dimension for opening/closing the gripper

This is **continuous control** -- the agent doesn't pick from a menu of discrete options, it outputs real numbers. This is harder than discrete action spaces (like Atari games where you pick "left" or "right").

### Reward (r)
A scalar number the environment gives the agent after each action, telling it how well it's doing. In Meta-World:
- Rewards are **dense** (you get a reward signal every step, not just at the end)
- Rewards are **shaped** (designed to guide the agent -- e.g., getting closer to the object gives you some reward even before you've picked it up)
- A total episode reward >= 500 means the task is solved

### Policy (pi)
The agent's strategy -- a function that maps states to actions. Written as pi(a|s): "given state s, what's the probability of taking action a?"

A **stochastic** policy outputs a probability distribution over actions (important for exploration). A **deterministic** policy always outputs the same action for a given state.

SAC uses a stochastic policy -- this is central to the paper.

## Episodes and Steps

Training happens in **episodes**. Each episode:
1. Environment resets to some initial state
2. Agent takes actions, one at a time
3. Each action produces: next state, reward, done flag
4. Episode ends after a fixed number of steps (500 in Meta-World) or when a termination condition is met

One action = one **step**. In our paper, we train for 1,000,000 (1M) steps total.

## The Return and Discounting

The agent doesn't just want to maximize immediate reward -- it wants to maximize the **return**, which is the total future reward:

```
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

**gamma** (discount factor, 0 to 1) controls how much the agent cares about future vs immediate reward:
- gamma = 0: only care about immediate reward (very shortsighted)
- gamma = 1: care equally about all future rewards (can be unstable)
- gamma = 0.99 (our paper): care a lot about future rewards but slightly prefer sooner ones

## The Goal of RL

Find a policy pi that maximizes expected return:

```
pi* = argmax_pi E[sum of discounted rewards when following pi]
```

That's it. Everything else -- Q-functions, policy gradients, SAC, entropy -- is machinery for solving this optimization problem efficiently.

## Why is RL Hard?

1. **Exploration vs exploitation**: Should the agent try something new (explore) or stick with what works (exploit)?
2. **Credit assignment**: When the robot finally picks up the object, which of the thousands of previous actions deserves credit?
3. **High-dimensional continuous spaces**: The agent must output precise real-valued actions, not pick from a small set.
4. **Sample inefficiency**: Learning from scratch in a physics simulator requires millions of interactions.
5. **Instability**: Neural networks approximating value functions can diverge or oscillate.

Our paper is fundamentally about problem #1 -- exploration vs exploitation -- and how SAC's mechanism for balancing them carries far more information than anyone realized.

---

**Next:** [02 -- Policy Gradients and Actor-Critic](02_policy_gradient_and_actor_critic.md)
