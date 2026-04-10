# 11 -- The Alpha-Bifurcation Phenomenon

This is the paper's core discovery.

## What We Found

When you train SAC with auto-entropy tuning on hard manipulation tasks and plot the alpha trajectory for each seed, the trajectories **split into distinct groups** -- they bifurcate. This happens early in training (200k-300k steps) and predicts eventual success or failure long before the reward curves diverge (400k-600k steps).

## The Three Regimes

### Stable Moderate (healthy)
- Alpha converges to the range 0.02-0.25
- Remains there throughout training
- Policy is actively exploring AND learning
- Corresponds to seeds that solve the task

### Collapse (failure)
- Alpha decays monotonically toward zero (< 0.005)
- Usually happens within the first 200k steps
- Policy becomes near-deterministic and stops exploring
- The seed never solves the task and never recovers
- Most common failure mode: 3/4 never-solved seeds on hard tasks

### Explosion (rare failure)
- Alpha grows unboundedly to 9+ (starting from ~1.0)
- The entropy term dominates the entire objective
- Policy is maximally random, can't commit to any action
- Only observed in 1/24 seeds on hard tasks
- The mirror image of collapse: collapse = too little entropy, explosion = too much

## Why It Happens: The Feedback Loop Hypothesis

### The virtuous cycle (leads to stable moderate alpha)
```
Seed gets lucky with initial random exploration
  -> Stumbles into some reward
    -> Q-values become informative (there's signal to learn from)
      -> Policy gradient points toward better actions
        -> Policy actively changes (improving)
          -> Changing policy = entropy stays near target
            -> Alpha stays moderate
              -> Exploration pressure is maintained
                -> Agent finds more reward
                  -> Positive spiral continues
```

### The collapse spiral (leads to alpha -> 0)
```
Seed's random exploration doesn't find reward
  -> Q-values have no informative signal
    -> Policy gradient is basically noise
      -> Policy barely changes
        -> Unchanging policy = entropy drifts below target
          -> Alpha decreases (the auto-tuner's response)
            -> Less exploration pressure
              -> Even less chance of finding reward
                -> Death spiral
```

### The explosion spiral (leads to alpha -> infinity)
```
Seed's policy happens to maintain chronically high entropy
  -> Entropy stays above target
    -> Alpha keeps increasing
      -> Entropy bonus dominates the objective
        -> Policy becomes even more random
          -> Still no reward (too random to accomplish anything)
            -> Q-values stagnate
              -> Nothing drives entropy below target
                -> Alpha increases further
```

## The Key Asymmetry

The critical insight is that **early task discovery creates a self-reinforcing positive cycle**, while **early failure creates a self-reinforcing negative cycle**.

Which cycle a seed enters is largely determined by early random exploration (the random seed controls initial network weights and early action sampling). This is why different seeds can have such dramatically different outcomes on the same task with the same hyperparameters.

Alpha records which cycle the seed entered. It's not causing the outcome -- it's a thermometer, not a thermostat (though it does participate in the feedback loop).

## Task Difficulty Dependence

The bifurcation only appears on hard tasks:

| Task | Difficulty | Bifurcation? | Why |
|------|-----------|-------------|------|
| drawer-close | Easy | No | Every seed finds reward quickly. All enter the positive cycle. |
| window-open | Medium | No | Same -- the task is easy enough that random exploration suffices. |
| door-open | Medium | No | Same. |
| peg-insert-side | Hard | Yes (1/12 fail) | Requires precision. Some seeds' random exploration doesn't find the insertion. |
| pick-place | Hard | Yes (6/12 fail) | Requires a multi-step sequence. Many seeds' random exploration fails. |

On easy tasks, the basin of attraction for success is so large that any initial random exploration falls into it. On hard tasks, the basin is small enough that some seeds miss it, triggering the collapse spiral.

## The Numbers

### Peg-insert-side (12 seeds)
- 11 solved: final alpha in [0.017, 0.177]
- 1 failed (collapse): final alpha = 0.001
- Clear separation between solved and failed alpha values

### Pick-place (12 seeds)
- 6 stably solved: final alpha in [0.071, 0.184]
- 3 discover-then-forget: final alpha in [0.026, 0.067] (alpha healthy, but policy degrades)
- 2 collapse: final alpha = 0.001-0.002
- 1 explosion: final alpha = 9.195

## Timing

The alpha bifurcation is visible at ~200k-300k steps. Reward divergence doesn't become clear until ~400k-600k steps. This 200k+ step lead time is what makes alpha useful as an **early warning diagnostic** -- you can see trouble coming before it shows up in the reward curve.

At 1M total training steps, seeing the problem 200k steps early means you can save ~20% of compute by stopping and restarting a doomed seed.

## Important Caveats

1. **Easy tasks also show low alpha at convergence.** A fully-solved, fully-converged policy has low entropy (it knows what to do and does it deterministically), so alpha naturally decays. The "healthy moderate range" describes the regime during *active learning*, not forever.

2. **Alpha is a necessary but not sufficient indicator.** Discover-then-forget seeds have healthy alpha but fail anyway. Alpha measures exploration health, not solution retention.

3. **The phenomenon requires reward signal separation.** If all seeds get similar rewards regardless of task progress (as in ManiSkill3), there's no asymmetry to drive bifurcation.

---

**Next:** [12 -- Three Failure Modes](12_three_failure_modes.md)
