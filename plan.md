# CoRL 2026 Battle Plan

**Deadlines:**
- Abstract: May 25, 2026 (53 days from today)
- Paper: May 28, 2026 (56 days from today)

**Paper thesis (revised):**
> SAC's auto-tuned entropy coefficient acts as an implicit success signal:
> α stays high on seeds that solve manipulation tasks and collapses on seeds that fail.
> Fixed entropy schedules destroy this adaptive mechanism, hurting both discovery and retention.

---

## Phase 1: New Experiments (April 2 – April 20)

### 1.1 — α Trajectory Recovery (HIGHEST PRIORITY)

You only have final-checkpoint α. You need the full trajectory across training.

**What to run:** Re-run Method A (SAC baseline) on peg-insert-side-v3 with α
logged every 1,000 steps.

**How many:** 8 seeds (same seeds 0–7 for direct comparison)

**Why only A on peg-insert?** This is the key figure. It shows α staying high
on solved seeds and collapsing on failed seeds ACROSS training, not just at the end.
One task, one method, one clean plot. Everything else is supporting evidence you
already have.

**Implementation — add this callback:**

```python
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

class AlphaTrajectoryCallback(BaseCallback):
    def __init__(self, log_path, log_every=1000):
        super().__init__()
        self.log_path = log_path
        self.log_every = log_every
        self.records = []

    def _on_step(self):
        if self.num_timesteps % self.log_every == 0:
            # SB3 stores log(α); recover α = exp(log_ent_coef)
            alpha = self.model.log_ent_coef.exp().item()
            self.records.append({
                'step': self.num_timesteps,
                'ent_coef': alpha
            })
        return True

    def _on_training_end(self):
        pd.DataFrame(self.records).to_csv(self.log_path, index=False)
```

**Estimated time:** 8 runs × 3-4 hours = ~1.5 days on your A6000

### 1.2 — Additional Tasks (REQUIRED for CoRL)

Two tasks is not enough for CoRL. You need 4-5 total. Add 3 more Meta-World v3 tasks.

**Which tasks to add:**

| Task | Why | Expected behavior |
|------|-----|-------------------|
| door-open-v3 | Medium difficulty, frequently benchmarked | A should solve 5-6/8 seeds |
| drawer-close-v3 | Easy task, SAC should solve most seeds | Controls for ceiling effects |
| window-open-v3 | Used in TemporalOT paper, direct comparison | Medium difficulty |

**What to run:** Method A (SAC baseline) only, 8 seeds each, 1M steps, WITH
the AlphaTrajectoryCallback from 1.1.

**Why only Method A?** Your story is about what SAC's auto-entropy does.
The 4-method ablation on peg-insert + pick-place already proves that fixed
schedules hurt. The new tasks just need to show the α-bifurcation pattern
generalizes. Running all 4 methods on 3 more tasks would be 96 extra runs
you don't have time for.

**Total new runs:** 3 tasks × 8 seeds = 24 runs
**Estimated time:** 24 × 3-4 hours ≈ 3-4 days (parallelize across tasks)

### 1.3 — Combined Compute Schedule

| Week | What | Runs | GPU-days |
|------|------|------|----------|
| Apr 2-6 | Peg-insert A with α trajectory (1.1) | 8 | 1.5 |
| Apr 6-12 | 3 new tasks, Method A, α trajectory (1.2) | 24 | 4 |
| Apr 12-14 | Buffer: re-run any crashed seeds | ~4 | 1 |
| **Total** | | **36 runs** | **~7 days** |

You can start writing the paper on April 7 while the new task runs are still going.

---

## Phase 2: Analysis (April 14 – April 22)

### 2.1 — The Key Figure: α Trajectory Bifurcation

From the 8 peg-insert runs with full α logging:

**Figure 1 (the paper's centerpiece):** Plot α vs training steps, one line per seed.
Color-code by final outcome: green for solved seeds (final reward ≥ 500),
red for unsolved seeds. You should see green lines staying at α ≈ 0.1-0.15
and red lines collapsing to α < 0.01.

**Then overlay a shaded band showing reward trajectory** on a secondary y-axis.
This creates the visual: "α high → reward rises → α stays high" vs
"α collapses → reward stays zero → α stays collapsed" — two attractors.

### 2.2 — Generalization: α-Success Correlation Across Tasks

From the 3 new tasks + 2 existing:

**Figure 2:** For each task, scatter plot of final α (x-axis) vs final reward (y-axis),
one point per seed. All 5 tasks on the same plot, different marker shapes per task.
You should see a clear positive correlation: high α → high reward.

Compute Spearman rank correlation across all 40 seed-task pairs (5 tasks × 8 seeds).
If ρ > 0.6 with p < 0.001, that's a strong finding.

### 2.3 — The Ablation (Already Done)

From your existing exp35 data:

**Figure 3:** The 4-method comparison on peg-insert + pick-place.
Learning curves with shaded CI. This shows that fixed annealing (B, D) hurts.
You already have this plot — just clean it up for the paper.

**Table 1:** Final reward mean ± std for all 4 methods × 2 tasks.
Include p-values for A vs B (the key comparison).

### 2.4 — Mechanistic Evidence (Already Done)

From your existing logs:

**Figure 4:** Buffer success fraction (peg-insert). Shows A accumulates
successful experience faster than B/D.

**Figure 5:** Q-value probe trajectories (pick-place). Shows B/D suffer
Q-value divergence from low entropy.

### 2.5 — Statistical Tests

For the α-bifurcation claim:
- Split seeds into solved/unsolved per task
- Mann-Whitney U test: final α of solved vs unsolved seeds
- Report across all 5 tasks

For A vs B:
- Welch t-test on final reward per task
- Report effect size (Cohen's d)

---

## Phase 3: Writing (April 7 – May 25)

### Paper Title

"Don't Override SAC's Entropy: Adaptive Temperature as an Implicit
Success Signal in Robotic Manipulation"

or shorter:

"SAC's Entropy Knows: Why Fixed Schedules Hurt Manipulation RL"

### Paper Structure (8 pages)

**1. Introduction (1 page)**

Opening: Entropy regularization in SAC is widely considered a lever that
practitioners should tune — recent work proposes annealing schedules, meta-learned
temperatures, and fixed coefficients. But does SAC's default auto-tuning already
work well for manipulation tasks?

Contribution: We show through a controlled 64-run ablation that (1) SAC's
auto-tuned α bifurcates into high-α (successful) and low-α (failed) regimes,
(2) this bifurcation is consistent across 5 manipulation tasks, and
(3) fixed entropy schedules force all seeds into the low-α failure regime,
reducing both sample efficiency and final performance.

Practical takeaway: Don't override SAC's entropy auto-tuning for dense-reward
manipulation tasks. The default is not just "good enough" — it's actively
adaptive in ways that fixed schedules cannot replicate.

**2. Related Work (0.75 page)**

- SAC and entropy regularization (Haarnoja et al. 2018)
- Entropy scheduling: TES-SAC (Zhou & Heywood, ICANN 2025), Meta-SAC
  (Wang & Ni 2020), fixed schedules in practice
- The explore-exploit trade-off in manipulation RL
- Negative results in RL: "Does RL Really Incentivize Reasoning?" (NeurIPS 2025
  best paper — legitimizes well-executed negative findings)

**3. Background (0.5 page)**

- SAC's dual optimization: policy reward + entropy bonus
- Auto-entropy: the log_ent_coef optimization with target entropy
- How α adapts: when policy entropy drops below target → α increases →
  more exploration. When entropy exceeds target → α decreases → more exploitation.

**4. Experimental Setup (1 page)**

- 2×2 ablation design: {env, env+demo} × {auto, annealed}
- 5 Meta-World tasks, 8 seeds per condition
- All hyperparameters
- Mechanistic logging: α trajectory, Q-probes, buffer composition

**5. Results (3 pages)**

5.1 — The α bifurcation (Figure 1 + Figure 2)
    Main finding: α stays high on successful seeds, collapses on failed ones.
    Correlation between final α and final performance.

5.2 — Fixed schedules hurt (Figure 3 + Table 1)
    A > B on peg-insert (3014 vs 2534) and pick-place (1525 vs 33).
    Annealing destroys the adaptive mechanism.

5.3 — Mechanistic explanation (Figure 4 + Figure 5)
    Buffer composition: annealed methods can't fill buffers with successes.
    Q-values: low entropy → Q-value divergence → policy collapse.

5.4 — Does demo reward help? (brief)
    Mixed: helps slightly on pick-place, hurts on peg-insert.
    Not the main story — relegate to 1 paragraph.

**6. Discussion (1 page)**

When might fixed annealing help? Possibly on sparse-reward tasks where
auto-entropy collapses before any discovery. Our results are on dense-reward
Meta-World tasks — we explicitly note this as a scope limitation.

Why does α bifurcate? Hypothesis: when the agent discovers successful
trajectories, the Q-values for those trajectories increase, which increases
the gradient signal for the policy, which keeps the policy improving, which
keeps policy entropy near target (because the policy is actively changing),
which keeps α high. Conversely, when the agent gets stuck, Q-values stagnate,
the policy stops improving, entropy drifts below target, and α collapses.
A virtuous cycle vs a death spiral.

Connection to "primacy bias" and "plasticity loss" literature — the α collapse
may be an early indicator of these phenomena.

**7. Limitations (0.25 page)**

- Dense-reward tasks only — results may differ on sparse reward
- Meta-World v3 only — needs validation on other benchmarks
- 8 seeds per condition — σ still high
- Final-checkpoint α for 3 new tasks (trajectory only for peg-insert)

**8. Conclusion (0.25 page)**

---

## Phase 4: Submission Polish (May 18 – May 28)

### May 18-22: Draft complete
- All figures finalized
- All numbers triple-checked against CSVs
- Limitations section honest and thorough

### May 22-25: Co-author review + polish
- Abstract refined (this matters a lot for CoRL)
- Supplementary video: show learning curves animating over time
  with α trajectory overlay

### May 25: Abstract submitted on OpenReview

### May 28: Full paper submitted

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| α trajectory doesn't show clean bifurcation | Low (final-checkpoint already shows it) | Plot anyway; even noisy trajectory with bifurcation at end is enough |
| New tasks don't show same pattern | Medium | If 3/5 tasks show it, that's sufficient; note exceptions in limitations |
| Reviewer says "only Meta-World" | High | Acknowledge in limitations; argue dense-reward manipulation is the primary use case for SAC |
| Reviewer says "not enough seeds" | Medium | 8 seeds × 5 tasks = 40 datapoints for correlation; argue this is standard for Meta-World |
| Reviewer says "we already knew auto-entropy is good" | Medium | Emphasize: nobody has shown the α-bifurcation or the Q-value divergence mechanism for WHY fixed schedules fail |

### Fallback: RA-L

If CoRL reviews come back negative (notification ~August), you have the
data ready for an RA-L submission immediately. RA-L is rolling, 6-month
turnaround, 6+2 pages. Same content, tighter presentation.

---

## Week-by-Week Calendar

| Week | Dates | What |
|------|-------|------|
| 1 | Apr 2-8 | Run peg-insert A α-trajectory (8 seeds). Start paper outline. |
| 2 | Apr 8-14 | Run 3 new tasks (24 seeds). Write Sections 1-3. |
| 3 | Apr 14-20 | Analysis: α-bifurcation figure, correlation plot. Write Section 5.1-5.2. |
| 4 | Apr 20-27 | Clean up existing mechanistic plots. Write Sections 5.3-6. Full draft done. |
| 5 | Apr 27 – May 4 | Internal review. Fix weak arguments. Rewrite intro. |
| 6 | May 4-11 | Polish figures to publication quality. Run any additional seeds if needed. |
| 7 | May 11-18 | Final writing pass. Supplementary materials. |
| 8 | May 18-25 | Abstract submitted May 25. Final polish. |
| 8+ | May 25-28 | Last-minute fixes. Submit May 28. |

---

## Checklist Before You Start

- [x] Implement AlphaTrajectoryCallback and test on 1 short run (10k steps)
- [x] Verify it produces correct α values by comparing to final_model.zip α (callback=0.028171, model=0.028162, match=True)
- [x] Set up Meta-World v3 for door-open, drawer-close, window-open (all confirmed in ML1.ENV_NAMES)
- [x] Prepare multi-seed launcher script (experiments/exp36_alpha_trajectory/launch_alpha_trajectory.sh)

