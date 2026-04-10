# 02 -- Policy Gradients and Actor-Critic Methods

## Two Families of RL

There are two main approaches to RL:

### Value-based methods (e.g., DQN)
Learn a **value function** that estimates how good each state (or state-action pair) is, then derive a policy from it. Example: "state A has value 10, state B has value 3, so go to A."

Problem: hard to use with continuous actions (you can't evaluate every possible real-valued action).

### Policy-based methods (e.g., REINFORCE)
Directly learn a **policy** -- a neural network that outputs actions. You optimize the policy parameters to maximize expected return.

Problem: high variance in gradient estimates; slow learning.

### Actor-Critic: The best of both
Combine both approaches:
- **Actor**: A neural network that outputs the policy (what action to take)
- **Critic**: A neural network that estimates value (how good the current situation is)

The critic tells the actor "that action was better/worse than expected," giving the actor a clearer learning signal. This reduces variance compared to pure policy gradients.

**SAC is an actor-critic method.**

## How the Actor Learns (Policy Gradient)

The actor is a neural network with parameters theta. We want to adjust theta so the policy produces higher-reward actions.

The policy gradient theorem gives us a direction to update:

```
gradient of J(theta) ~ E[gradient of log pi(a|s) * advantage]
```

In plain English: "Make actions that turned out better than expected more likely, and actions that turned out worse less likely."

The **advantage** is: "how much better was this action than what I expected?" This is where the critic comes in -- it provides the baseline expectation.

## How the Critic Learns (Q-function)

The critic learns a Q-function Q(s, a) that answers: "If I'm in state s and take action a, then follow my policy forever, what's my expected return?"

It learns by minimizing the **temporal difference (TD) error**:

```
TD error = Q(s, a) - [r + gamma * Q(s', a')]
```

This says: "My estimate Q(s,a) should equal the actual reward r plus my discounted estimate of the next state." When this error is large, the critic's estimate was wrong and needs updating.

## The Actor-Critic Loop

```
1. Agent is in state s
2. Actor outputs action a = pi(s)
3. Environment returns reward r and next state s'
4. Critic evaluates: "Was this action good?" using Q(s, a)
5. Critic updates its Q-estimate using TD error
6. Actor updates its policy to favor actions the critic says are good
7. Repeat
```

The actor and critic improve each other: the critic gets better at evaluating, and the actor gets better at choosing actions.

## Stochastic vs Deterministic Actors

**Deterministic actor** (like DDPG, TD3): outputs a single action for each state.
- Pro: Lower variance, simpler.
- Con: No built-in exploration. You need to add noise artificially (e.g., Gaussian noise on the action).

**Stochastic actor** (like SAC): outputs a probability distribution over actions, then samples from it.
- Pro: Natural exploration -- randomness is built into the policy.
- Con: Slightly more complex.

SAC uses a stochastic actor with a **squashed Gaussian** distribution:
1. The network outputs a mean (mu) and standard deviation (sigma) for each action dimension
2. Sample from Normal(mu, sigma)
3. Apply tanh to squash the output to [-1, 1] (the valid action range)

This is critical: the randomness in SAC's policy is not just noise -- it's a fundamental part of the algorithm. The **entropy** of this distribution is what alpha controls, and what our paper analyzes.

## Why Actor-Critic for Robotics?

For continuous-control robotics, actor-critic methods dominate because:
1. They handle continuous action spaces naturally (the actor outputs real numbers)
2. They're more sample-efficient than pure policy gradient methods (the critic provides a lower-variance learning signal)
3. Off-policy learning is possible (we can reuse past experience; see the replay buffer explainer)

SAC added one more ingredient to make this work even better: **entropy regularization**. That's next.

---

**Next:** [03 -- Q-Functions and the Overestimation Problem](03_q_functions_and_overestimation.md)
