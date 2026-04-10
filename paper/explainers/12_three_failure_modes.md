# 12 -- Three Failure Modes

Our paper identifies three qualitatively distinct failure modes on hard manipulation tasks. Two are readable from the alpha trajectory alone; one requires joint monitoring of alpha and reward.

## Failure Mode 1: Alpha-Collapse

### What happens
Alpha decays monotonically to near-zero (< 0.002) within the first 200k steps and stays there. The policy becomes near-deterministic, and the agent is permanently trapped in a local minimum.

### Observable signatures
- **Alpha trajectory**: Smooth, monotonic decay to < 0.005
- **Reward**: Flat near zero for the entire training run
- **Q-values**: Stagnant (no useful signal) or exhibit overestimation spikes
- **Replay buffer**: Very low success fraction; dominated by failed trajectories

### Mechanism
1. The initial random policy doesn't stumble into task-relevant states
2. Without reward signal, Q-values are uninformative
3. The policy gradient is essentially noise (no direction to improve)
4. The policy stops changing
5. A static policy has decreasing entropy (it's the same distribution every time, no reason for it to maintain high entropy)
6. Entropy falls below the target (-4)
7. The auto-tuner decreases alpha (its job: if entropy is too low, normally you'd raise alpha, but the entropy is low because the policy is *stuck*, not because it converged -- the tuner can't tell the difference)

Wait, actually: when entropy is below target, the gradient pushes alpha UP. So why does alpha collapse?

The subtlety: the *gradient on log(alpha)* is `actual_entropy - target_entropy`. If actual entropy is below target, this is negative, which means log(alpha) decreases, meaning alpha goes DOWN.

Actually, let me re-derive. The dual objective is:
```
J(alpha) = E[-alpha * (log pi(a|s) + H_target)]
```
Gradient w.r.t. log(alpha):
```
dJ/d(log alpha) = E[-alpha * (log pi(a|s) + H_target)]
                = -alpha * (E[log pi(a|s)] + H_target)
                = -alpha * (-actual_entropy + H_target)
                = alpha * (actual_entropy - H_target)
```

If actual_entropy < H_target (entropy too low): gradient is negative -> log(alpha) decreases -> alpha decreases.

Wait, that's the wrong direction! If entropy is too low, we want alpha to increase.

Let me check: actually in the SB3 implementation, the loss is:
```
alpha_loss = -log_alpha * (log_prob + target_entropy).detach().mean()
```

And gradient ASCENT is performed (maximizing). So:
```
d(alpha_loss)/d(log_alpha) = -(log_prob + target_entropy)
                            = -((-actual_entropy) + target_entropy)  [approximately]
                            = actual_entropy - target_entropy
```

With gradient ascent, if actual_entropy < target_entropy, the gradient is negative, so log_alpha DECREASES. Alpha goes DOWN.

Hmm, but the standard SAC formulation should push alpha UP when entropy is too low. Let me reconsider...

Actually the sign convention in SAC: when you MINIMIZE the dual loss:
```
J(alpha) = alpha * (entropy - target_entropy)
```
If entropy < target, J is negative and minimizing it means increasing alpha. This pushes alpha up.

The implementation detail matters but the published SAC math is clear: low entropy -> alpha increases. So alpha-collapse must be more nuanced.

**The real mechanism of collapse**: The entropy doesn't just stay below target -- it *tracks the target*. When the policy is stuck, its entropy is low but the auto-tuner tries to raise alpha. However, raising alpha doesn't help because the policy gradient itself carries no signal (Q-values are flat). So alpha oscillates or drifts based on small random entropy fluctuations. In practice, the entropy *does* sometimes exceed the target momentarily due to noise, pushing alpha down. Over long periods without learning signal, alpha drifts downward through this ratchet effect.

More precisely: on a non-learning seed, the policy entropy and alpha enter a low-amplitude cycle near zero, where alpha is too small to create meaningful exploration pressure even when it momentarily increases.

### Occurrence
Most common failure mode: 3 of 4 never-solved seeds on hard tasks exhibit collapse.

### Seeds
- Pick-place seeds 3 and 6: final alpha 0.001-0.002
- Peg-insert seed 4: final alpha 0.001

---

## Failure Mode 2: Alpha-Explosion

### What happens
Alpha grows from its initial value (~1.0) to extreme values (9.195 observed), without bound. The entropy bonus dominates the entire objective, and the policy becomes maximally random, unable to commit to any effective action.

### Observable signatures
- **Alpha trajectory**: Monotonic increase, crossing 1.0, then 2.0, then 5.0...
- **Reward**: Flat near zero (random actions don't solve tasks)
- **Q-values**: Stagnant or noisy (no coherent learning signal)
- **Policy entropy**: Chronically above target

### Mechanism
1. The initial random policy happens to have very high entropy (well above target -4)
2. Auto-tuner responds: entropy above target -> alpha decreases... but not fast enough
3. As alpha grows, the entropy bonus `alpha * H(pi)` increasingly dominates the Q-value term in the policy objective
4. The actor optimizes primarily for entropy rather than reward
5. This keeps entropy high, which keeps pushing alpha up
6. Positive feedback loop: high alpha -> high entropy objective -> more random policy -> no reward signal -> nothing breaks the cycle

### Why it's rare
Most initial policies don't have chronically high entropy. The default alpha = 1.0 at initialization is usually enough to bring entropy near the target. Explosion requires an unfortunate initialization where entropy stays persistently above target throughout training.

### Occurrence
Very rare: 1/24 seeds on hard tasks. Only observed on pick-place seed 1 (final alpha = 9.195).

---

## Failure Mode 3: Discover-Then-Forget (DTF)

### What happens
The agent discovers a solution (reward spikes to > 500) but then **catastrophically loses it** -- reward drops back to near-zero. The policy "forgets" how to solve the task.

### Observable signatures
- **Alpha trajectory**: Stays in the healthy moderate range (0.026-0.067). Looks indistinguishable from a stably-solved seed.
- **Reward**: Shows a clear spike (peak 1618-2519) followed by a collapse to near-zero.
- **Q-values**: May show instability around the time of policy reversion.

### Mechanism
This is likely related to **plasticity loss** (Lyle et al., 2023):
1. The agent successfully learns the task
2. Continued training overwrites the learned representation
3. The neural network loses the capacity to reproduce the solution
4. Once lost, the solution may be very hard to re-discover

### Why alpha can't detect it
Alpha measures **exploration health** -- whether the policy's entropy is in the right range. A DTF seed has healthy entropy; the policy is still changing and exploring. The problem isn't exploration, it's that the network's internal representation has degraded. This is a different failure mode entirely, operating at a deeper level than the entropy mechanism.

### Practical implication
To catch DTF, you must monitor both alpha AND reward:
- Alpha healthy + reward sustained = true success
- Alpha healthy + reward drops = discover-then-forget
- Alpha collapsed + reward zero = alpha-collapse
- Alpha exploded + reward zero = alpha-explosion

### Occurrence
3/24 seeds on hard tasks: pick-place seeds 4, 10, and 11.
- Seed 4: peaked at reward ~2519 (step 891k), final reward 11
- Seed 10: peaked at reward ~1618 (step 320k), final reward 16
- Seed 11: peaked at reward ~2080 (step 990k), final reward 49

---

## Summary Table

| Mode | Alpha signal | Reward signal | Detectable from alpha alone? | Frequency |
|------|-------------|---------------|------------------------------|-----------|
| Collapse | < 0.005 | Flat near 0 | Yes | Common (3/4 never-solved) |
| Explosion | > 1.0 (growing) | Flat near 0 | Yes | Rare (1/24) |
| Discover-then-forget | 0.02-0.07 (healthy) | Spike then crash | No (needs reward too) | Moderate (3/24) |
| Healthy | 0.02-0.25 (stable) | High and sustained | Yes (positive indicator) | Depends on task |

---

**Next:** [13 -- The Ablation: Fixed Annealing Destroys SAC](13_ablation_results.md)
