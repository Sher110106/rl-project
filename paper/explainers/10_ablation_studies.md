# 10 -- Ablation Studies and Experimental Design

## What is an Ablation Study?

An ablation study is an experiment where you **systematically remove or change one component** of a system to understand its contribution. The term comes from neuroscience, where researchers would ablate (destroy) specific brain regions to understand their function.

In machine learning:
- You have a system that works (e.g., SAC with auto-entropy tuning)
- You remove or replace one component (e.g., replace auto-tuning with a fixed schedule)
- You compare performance to the original
- If performance drops, that component was important ("load-bearing")

This is the standard way to establish **causal claims** in ML research, as opposed to just observational claims.

## Observational vs Causal Claims

### Observational (correlation)
"We observed that seeds with low alpha tend to fail."

This is what our observational study (Exp. 36) establishes. It's valuable but doesn't prove that low alpha *causes* failure. Maybe failure causes low alpha. Maybe both are caused by some third factor.

### Causal (ablation)
"When we force alpha to be low via a fixed schedule, seeds fail."

This is what our ablation study (Exp. 35) establishes. By actively intervening (changing the entropy mechanism), we show that the auto-tuning is causally responsible for success on hard tasks.

The combination of observation + ablation makes for a much stronger argument than either alone.

## Our Experimental Design

### Exp. 36: Observational Study (64 runs)
- 5 Meta-World tasks (easy to hard)
- 8 seeds for easy/medium tasks, 12 seeds for hard tasks
- Standard SAC with auto-entropy tuning
- Purpose: Observe and characterize alpha dynamics across seeds and tasks
- Claim supported: Alpha trajectories bifurcate and predict outcome

### Exp. 35: Causal Ablation (64 runs)
- 2 hard tasks only (peg-insert, pick-place)
- 4 conditions x 8 seeds each
- 2x2 factorial design
- Purpose: Establish that auto-tuning is causally important
- Claim supported: Removing auto-tuning destroys performance

## The 2x2 Factorial Design

A factorial design tests multiple factors simultaneously, which is more efficient than testing one at a time. Our factors:

**Factor 1: Entropy method**
- Auto-tuned (SAC default)
- Fixed annealing (alpha: 0.1 -> 0.005 over 500k steps)

**Factor 2: Reward**
- Standard environment reward
- Standard reward + self-bootstrapped demo reward (k-NN bonus)

This gives four conditions:

| | Auto-entropy | Fixed annealing |
|---|---|---|
| **Standard reward** | Method A (baseline) | Method B |
| **Demo reward** | Method C | Method D |

### Why this design?

1. **Method A vs B**: Does removing auto-tuning hurt? (Yes, catastrophically on pick-place)
2. **Method C vs D**: Does adding a demo reward rescue the annealing failure? (No)
3. **Method A vs C**: Does the demo reward help? (No, it actually hurts slightly)
4. **Interaction effects**: Does the combination of demo + auto help more than either alone? (No)

The 2x2 design lets us see all of this from 64 runs, rather than needing separate experiments for each comparison.

## Why Multiple Seeds?

In RL, results are **extremely sensitive to random seed**. The same algorithm with the same hyperparameters can succeed brilliantly or fail completely depending on:
- Initial network weights (random)
- Order of experience collection (random)
- Action sampling (random)

This is why our paper uses 8-12 seeds per condition. A single seed proves nothing. The distribution across seeds is what matters.

Following Agarwal et al. (2021) -- who showed that many RL papers had unreliable results due to too few seeds -- we report:
- Per-seed results (every individual seed's outcome)
- Solve rates (fraction of seeds that succeed)
- Mean +/- std of final reward
- Statistical tests (Mann-Whitney U, permutation test)

## The Annealing Schedule

The fixed annealing schedule we test:
```
alpha(t) = 0.1 - (0.1 - 0.005) * min(t/500000, 1.0)
```

- Starts at alpha = 0.1
- Linearly decreases to alpha = 0.005 over 500k steps
- Stays at 0.005 for the remaining 500k steps

This is a reasonable schedule -- 0.1 is in the range that solved seeds naturally converge to. But "reasonable" isn't good enough. The problem is that it's not adaptive:
- If the agent hasn't discovered the task by 500k steps, forcing alpha to 0.005 kills any remaining exploration
- If the agent needs more or less exploration than the schedule provides, tough luck

## The Demo Reward

Methods C and D add a supplementary reward signal:
- For each transition, find the k=5 nearest neighbors among successful transitions in the replay buffer
- Give a bonus proportional to distance (closer to past successes = more bonus)
- Parameters: k=5, sigma=0.30, scale=0.5

The idea was that this "self-bootstrapped" reward might help struggling seeds by rewarding them for being in states similar to past successes. The result: it didn't help and slightly hurt. The k-NN bonus introduces additional noise into the Q-function's optimization, disrupting the delicate entropy feedback loop.

This is itself an interesting finding: auxiliary reward shaping can *harm* rather than help, because it interferes with SAC's adaptive mechanism.

## How to Read the Results

When looking at the ablation results:
- **Solve rate**: Most important metric. How many seeds ever solve the task?
- **Mean final reward**: Average reward at the end of training. High variance indicates some seeds solved and some didn't.
- **Alpha trajectories**: Show what the entropy mechanism is doing under each condition.
- **Q-value probes**: Show whether the critic is healthy or pathological.
- **Buffer success fraction**: Shows how quickly successful experience accumulates.

The killer finding: on pick-place, Method A (auto-entropy) solves 4/8 seeds with mean reward 1525. Method B (annealing) solves 0/8 seeds with mean reward 33. That's a 98% reduction in performance from removing a single mechanism.

---

**Next:** [11 -- The Alpha-Bifurcation Phenomenon](11_alpha_bifurcation.md)
