# 14 -- Alpha as a Training Diagnostic

This is the paper's practical contribution: how to use alpha monitoring in real training workflows.

## The Classification Result

On the two hard tasks (24 seeds total from Exp. 36), alpha correctly classifies all seeds by outcome:

### Two-sided threshold: 0.005 < alpha < 1.0

| Predicted | alpha < 0.005 | 0.005 < alpha < 1.0 | alpha > 1.0 |
|-----------|---------------|----------------------|-------------|
| Outcome   | Never-solved (collapse) | Ever-solved | Never-solved (explosion) |
| Accuracy  | 3/3 correct | 20/20 correct | 1/1 correct |

**24/24 = 100% accuracy** using a simple two-sided threshold.

### What about the single-sided threshold?
Using only alpha_low = 0.005:
- 23/24 correct (96%)
- Misses the 1 explosion seed (alpha = 9.195 is above 0.005, so it's classified as "healthy" by the single-sided rule)

The two-sided rule catches both collapse AND explosion.

### What about discover-then-forget?
The 3 DTF seeds (pick-place 4, 10, 11) have final alpha in [0.026, 0.067] -- well within the healthy range. Alpha correctly classifies them as **ever-solved** (they did solve the task during training). But they're not solved at the end.

To detect DTF, you need to monitor both alpha and reward. If alpha is healthy but reward drops after a previous peak, that's DTF.

## Practical Monitoring Protocol

### What to log
Add alpha logging to your SAC training:
```python
# In your callback, every 1000 steps:
alpha = exp(model.log_ent_coef.item())  # SB3
```

This is essentially free -- alpha is already computed during training.

### When to check
Monitor alpha at mid-training (~300k-400k steps for a 1M step run):

### Decision tree
```
Is alpha < 0.005?
  -> YES: Alpha-collapse. Seed is likely stuck permanently.
          Consider early termination and restart with new seed.
  -> NO: Continue...

Is alpha > 1.0?
  -> YES: Alpha-explosion. Seed is diverging.
          Consider early termination and restart with new seed.
  -> NO: Alpha is healthy. Continue training.

If alpha is healthy at end of training, check reward:
  Is reward sustained above success threshold?
    -> YES: True success.
    -> NO: Discover-then-forget. Policy lost its solution.
```

### Compute savings
If you detect collapse/explosion at 300k-400k steps in a 1M step run, you can terminate early and restart, saving **60-70% of remaining compute**. Instead of wasting 600k more steps on a doomed seed, you start a fresh seed that might succeed.

For a 12-seed experiment at 1M steps each:
- Without monitoring: 12M steps total
- With monitoring: ~12M steps (same for the good seeds) but bad seeds terminate early, saving ~3-4M steps
- That's a ~25-30% compute reduction with no loss of information

## What Alpha Can Tell You

| Alpha value/trend | Interpretation | Action |
|---|---|---|
| 0.02-0.25 and stable | Active learning, exploration healthy | Continue training |
| Monotonically decreasing toward 0 | Collapse forming | Monitor closely; terminate if < 0.005 |
| Monotonically increasing above 1 | Explosion forming | Terminate early, restart |
| Oscillating in moderate range | Normal variation | Continue |
| Low after convergence on easy task | Task is fully solved | This is normal; don't confuse with collapse |

## What Alpha Cannot Tell You

1. **Solution retention**: Alpha looks the same for stably-solved and DTF seeds. You need reward monitoring for this.

2. **Absolute performance**: Two seeds can both have healthy alpha but achieve very different final rewards. Alpha is binary (healthy vs unhealthy), not a performance predictor.

3. **Sparse-reward tasks**: If no seeds get any reward signal, alpha may behave differently. Our findings are validated on dense-reward tasks only.

4. **Different reward structures**: On ManiSkill3 (uniformly shaped rewards), alpha didn't bifurcate at all. The diagnostic works when the reward landscape creates clear success/failure separation.

## Recommended Dashboard

For any SAC training run, plot:
1. **Alpha trajectory** (top panel): should be the first thing you look at
2. **Reward curve** (middle panel): confirms whether the agent is actually succeeding
3. **Replay buffer success fraction** (bottom panel): shows how quickly good experience accumulates

Alpha and reward together catch all three failure modes:
- Collapse: alpha collapses, reward flat
- Explosion: alpha explodes, reward flat
- DTF: alpha healthy, reward spikes then drops

## Future Directions

The paper ends with an open question: **Could we intervene based on alpha?**

Instead of just monitoring alpha and early-stopping, could we:
- Detect collapse forming and temporarily boost exploration?
- Detect explosion forming and dampen the entropy mechanism?
- Create an "alpha-aware" schedule that responds to the signal rather than replacing it?

This is the difference between a diagnostic (tells you something is wrong) and a treatment (fixes it). Our paper establishes the diagnostic; the treatment is future work.

---

## You've Reached the End

If you've read all 14 documents, you now understand:
- The RL foundations (states, actions, rewards, policies)
- The actor-critic architecture
- Q-functions and their pathologies
- Replay buffers and off-policy learning
- Entropy and the maximum entropy framework
- SAC: how it works and why it's used
- Auto-entropy tuning: the alpha mechanism and its feedback loop
- Meta-World: the benchmark and its reward structure
- Prior failure modes (primacy bias, plasticity loss)
- Ablation methodology
- **The alpha-bifurcation** (our core finding)
- **Three failure modes** (collapse, explosion, DTF)
- **Why fixed annealing destroys SAC** (the causal evidence)
- **How to use alpha as a diagnostic** (the practical contribution)

You should be able to explain the full paper, from "what is SAC?" all the way to "why does a two-sided threshold on alpha predict success with 100% accuracy on hard tasks."
