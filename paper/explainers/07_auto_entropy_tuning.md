# 07 -- Auto-Entropy Tuning (the Alpha Mechanism)

This is the most important background concept for the paper. Everything we discover revolves around how alpha behaves.

## The Problem: How to Set Alpha?

Recall that SAC's objective is:
```
J(pi) = E[rewards + alpha * entropy]
```

Alpha controls exploration vs exploitation. But what value should it be?

- Too high -> agent is too random, never learns
- Too low -> agent is too deterministic, gets stuck in local optima
- Just right -> agent explores enough to find solutions, then exploits them

The "right" value of alpha isn't fixed -- it should change over training. Early on, the agent knows nothing and should explore widely (high alpha). As it learns, it should gradually focus on what works (lower alpha). But how much? And when?

## Approach 1: Fixed Alpha
Just pick a value (e.g., alpha = 0.2) and keep it constant. Simple, but you're guessing. Works for some tasks, fails for others.

## Approach 2: Annealing Schedule
Start with a high alpha and decrease it linearly over training:
```
alpha(t) = alpha_start - (alpha_start - alpha_end) * t / T
```

Example from our ablation: alpha: 0.1 -> 0.005 over 500k steps.

This is what many practitioners do. Our paper shows it can be catastrophic.

## Approach 3: Auto-Tuning (What SAC v2 Does)

Instead of setting alpha directly, set a **target entropy** and let alpha learn automatically.

### The Constrained Optimization View

Auto-tuning frames the problem as constrained optimization:

```
Maximize: E[sum of rewards]
Subject to: H(pi(.|s_t)) >= H_target  for all t
```

"Maximize reward, but the policy must maintain at least H_target entropy."

This is solved via **Lagrangian relaxation**. The Lagrangian is:
```
L = E[rewards] + alpha * (E[H(pi)] - H_target)
```

Alpha is the **Lagrange multiplier** -- a variable that the optimization automatically adjusts to enforce the constraint. If entropy is too low (constraint violated), alpha increases to push entropy up. If entropy is too high (constraint slack), alpha decreases.

### The Dual Objective

In practice, SAC optimizes the dual:
```
J(alpha) = E[-alpha * (log pi(a|s) + H_target)]
```

Taking the gradient with respect to log(alpha):
```
gradient = E[-(log pi(a|s) + H_target)]
         = E[-log pi(a|s)] - H_target
         = actual_entropy - H_target
```

So:
- **If actual entropy < H_target** (policy too deterministic): gradient is negative, log(alpha) increases, alpha goes UP -> more exploration pressure
- **If actual entropy > H_target** (policy too random): gradient is positive, log(alpha) decreases, alpha goes DOWN -> less exploration pressure

The optimization is over **log(alpha)** (not alpha directly) to ensure alpha stays positive.

### Target Entropy

The target entropy in Stable-Baselines3 defaults to:
```
H_target = -dim(action_space)
```

For Meta-World (4D action space): H_target = -4

This heuristic means "maintain about 1 nat of entropy per action dimension." The negative sign is because continuous action log-probabilities are typically negative.

### Initial Alpha

SAC starts with log(alpha) = 0, so alpha = exp(0) = 1.0 initially. From there, it adjusts based on the policy's actual entropy.

## The Feedback Loop

This is the critical insight that our paper builds on. Auto-tuning creates a feedback loop between the policy and alpha:

### Healthy Loop (leads to success)
```
Agent discovers reward
  -> Q-values become informative
    -> Policy gradient is useful
      -> Policy actively changes
        -> Entropy stays near target (changing policy = entropy)
          -> Alpha stays moderate
            -> Exploration continues
              -> Agent discovers more reward
```

### Collapse Loop (leads to failure)
```
Agent fails to discover reward
  -> Q-values stagnate (no informative signal)
    -> Policy gradient is uninformative
      -> Policy stops changing
        -> Entropy drops below target
          -> Alpha drops toward 0
            -> Exploration pressure vanishes
              -> Agent NEVER discovers reward
```

### Explosion Loop (rare, leads to failure)
```
Policy entropy stays chronically above target
  -> Alpha keeps increasing
    -> Entropy bonus dominates the objective
      -> Policy becomes even more random
        -> Agent can't commit to any action
          -> No reward signal
            -> Alpha keeps increasing
```

## Why Auto-Tuning is Load-Bearing

The feedback loop means alpha isn't just a hyperparameter -- it's an adaptive control mechanism. It increases exploration when the agent is stuck and decreases it when the agent is learning.

When you replace this with a fixed schedule (like in our ablation), you break the feedback:
- The schedule forces alpha to 0.005 regardless of whether the agent has learned anything
- If the agent hasn't discovered the task by 500k steps, it now has almost zero exploration pressure
- Without exploration, it will never discover the task
- Game over

This is exactly what happens: 0/8 seeds solve pick-place under annealing vs 4/8 with auto-tuning.

## What Alpha Tells You

Because alpha responds to the policy's learning dynamics, it encodes information:

| Alpha range | What it means | Prognosis |
|-------------|---------------|-----------|
| 0.02 - 0.25 | Policy is actively learning, entropy near target | Healthy |
| < 0.005 | Policy stopped changing, entropy collapsed | Likely stuck permanently |
| > 1.0 | Entropy above target, exploration dominating | Likely diverging |
| ~ 1.0 (initial) | Training just started | Too early to tell |

This is the basis for using alpha as a training diagnostic (our paper's practical contribution).

## Connection to the Paper's Title

"SAC's Entropy Coefficient as an Implicit Success Signal" -- alpha was designed to be a stability mechanism (keep entropy near target). We show it's inadvertently also a success signal (its trajectory predicts whether training will succeed or fail). This "implicit" diagnostic is free -- it's already computed during training, you just need to log and monitor it.

---

**Next:** [08 -- Meta-World Benchmark](08_meta_world_benchmark.md)
