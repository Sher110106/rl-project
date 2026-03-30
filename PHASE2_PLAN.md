# Phase 2: Stabilization Plan
## From Discovery to Retention — Focused Direction for Publication

**Written:** 2026-03-28
**Based on:** 34 experiments (exp1–exp34), 3 environments, 5 reward variants
**Status:** All Wave 1–4 experiments complete. Pivoting to stabilization.

---

## 1. What the Data Actually Proves

After 34 experiments across PyBullet and Meta-World, we have three clean empirical facts:

### Fact 1: Demo reward = high discovery, low retention

| Experiment | Env | Peak Reward | Last-5 Mean | Gap (peak / last-5) |
|---|---|---|---|---|
| Exp19 (OT demo, pre-loaded buffer) | PyBullet | 310.9 | 196.2 | 1.6× — **stable** |
| Exp31 demo_only pick-place | Meta-World | 834.1 | 4.4 | **190×** — collapse |
| Exp23 CASID peg-insert | Meta-World | 4547.2 | 9.3 | **489×** — collapse |
| Exp33 demo_only peg-insert | Meta-World | 91.3 | 3.6 | 25× — no retention |

**Pattern:** The method discovers solutions (huge peaks) but forgets them within a few eval cycles. The ONE exception is PyBullet exp19 — where the demo buffer was **pre-loaded** with 10k states from prior successful episodes.

**Root cause:** On Meta-World, the demo buffer starts empty and must be populated from scratch. The chicken-and-egg problem: need successes to fill the buffer → need the buffer to guide toward success. By the time enough demos exist, the policy has moved on.

### Fact 2: Causal reward = task-dependent, not general

| Task | Causal helps? | Why |
|---|---|---|
| PyBullet pick-place | **Yes — breakthrough** (72% → 92%) | Base reward was weak/sparse; causal was primary signal |
| Meta-World push-v3 | Marginal (peak 1066, sustained 11) | Push = interaction-heavy, correlation is relevant |
| Meta-World pick-place-v3 | No (peak 16.5, sustained 5.1) | Meta-World already shapes toward grasping |
| Meta-World peg-insert-v3 | No (peak 2088, sustained 9.1) | Precise insertion ≠ gripper correlation |

**The lesson:** Causal reward is powerful when the base reward is sparse/weak (PyBullet). It's redundant when the base reward is already well-designed (Meta-World). This is actually a useful finding — but it means causal is NOT the paper's core contribution.

### Fact 3: Baseline SAC is surprisingly strong on Meta-World

| Task | Baseline SAC Last-5 | Best CASID variant Last-5 | Who wins? |
|---|---|---|---|
| pick-place-v3 | 4.8 | 29.6 (no_filter) | CASID variant (6×) |
| push-v3 | 15.7 | 31.5 (demo_only) | Demo (2×) |
| peg-insert-v3 | **828.7** | 6.6 (no_filter) | **Baseline SAC (125×)** |

**The uncomfortable truth on peg-insert:** Baseline SAC found the task and stayed there. Every CASID variant spiked higher but crashed. This is a single-seed result — we don't know if SAC's peg-insert success is robust or lucky. That's why multi-seed runs are essential.

---

## 2. The Core Failure Mode

The data points to one specific failure mode:

**Explore → Discover → Forget**

The bottleneck is NOT:
- ❌ Exploration (peaks prove the agent CAN find solutions)
- ❌ Reward richness (Meta-World already has excellent reward shaping)
- ❌ Causal reasoning (redundant on dense benchmarks)

The bottleneck IS:
- ✅ **Policy stabilization after first success**
- ✅ **Demo buffer bootstrap on cold-start environments**
- ✅ **Non-stationary reward confusing the critic** (r_demo changes as buffer grows)

### Why does SAC forget?

Three interacting causes:

1. **SAC's entropy bonus fights exploitation.** After finding peg-insertion, the `α · entropy` term pushes the policy to explore alternatives. On Meta-World's well-shaped reward, this doesn't matter — the reward keeps pulling the policy back. But CASID's extra terms create a noisier reward landscape that the entropy term destabilizes.

2. **The demo buffer bootstrapping is broken on cold-start.** In PyBullet exp19 (our best result at 90%), we pre-loaded the buffer with 10k states from prior experiments. On Meta-World, the buffer starts empty. This is the structural reason PyBullet worked and Meta-World doesn't.

3. **Non-stationary reward confuses the Q-function.** Every time the demo buffer grows, `r_demo` changes. The critic's Q-estimates become inconsistent → actor oscillates → peaks followed by crashes.

---

## 3. What NOT To Do Next

Based on the data, these directions are dead ends:

| ❌ Don't do | Why |
|---|---|
| More reward terms | Problem is retention, not reward quality |
| Tuning causal scale | Causal is task-specific, not the paper's core |
| True NN Granger (exp16-style) | Failed at 1M steps, too slow to warm up |
| Q-UCB potential shaping | Marginal gain (24%), complex to implement |
| Running CASID (full) with all components | Demo_only is the cleaner story |
| More ablations on seed 0 | Single seed is meaningless; need multi-seed |
| Optimizing for peak reward | Paper requires sustained performance (last-N mean) |

---

## 4. The Focused Plan

### Phase 1 — Fast validation (12 runs)

**Goal:** Establish whether demo_only vs SAC shows a real signal direction across seeds.

| Method | Tasks | Seeds | Runs |
|---|---|---|---|
| demo_only | pick-place-v3, peg-insert-v3 | seeds 1, 2, 3 | 6 |
| baseline SAC | pick-place-v3, peg-insert-v3 | seeds 1, 2, 3 | 6 |

**Why these tasks only:**
- push-v3 is already understood (causal dominates, not our story)
- pick-place = core claim (grasping)
- peg-insert = hardest task (precision), highest-variance results, must resolve SAC's single-seed dominance

**Why 3 seeds (not 5):**
- Enough to detect signal direction (is demo_only systematically better/worse?)
- If signal is clear → proceed to Phase 3 with 5 seeds
- If signal is ambiguous → stabilization experiments first (Phase 2)

**What we measure:** Last-20 eval mean, not peaks. A method that peaks at 4000 but averages 10 is worse than one that peaks at 200 but averages 150.

**Already have:** seed 0 for both methods on both tasks (exp21, exp31, exp33, exp34). So Phase 1 adds seeds 1-3.

### Phase 2 — Stabilization experiments (4–8 runs)

**Goal:** Test two mechanisms that convert "discovery" into "retention."

**Run ONLY if Phase 1 shows promise** (demo_only has higher peaks but lower sustained → stabilization can help).

#### Experiment A: Smoothed demo reward

The current demo reward uses k=1 nearest neighbour, which is spiky. A single outlier demo state can dominate the signal.

**Changes:**
- `k = 5` nearest neighbors (average distance → smoother gradient)
- `OT_SIGMA = 0.30` (from 0.15 — wider neighborhood, less binary)
- `OT_SCALE = 0.5` (from 0.3 — stronger anchor toward demo states)

**Why this might work:** The smoother reward provides a wider basin of attraction around successful states. Instead of "be exactly here" (spiky k=1), it says "be anywhere near here" (smooth k=5). This resists the entropy-driven policy drift that causes forgetting.

**Runs:** 2 tasks × 1-2 seeds = 2-4 runs

#### Experiment B: Replay biasing (SIL-lite)

When the agent succeeds, duplicate those transitions in the replay buffer. This biases SAC's replay sampling toward successful experiences, counteracting the FIFO washout.

**Implementation:** Override SB3's `_store_transition`. When `info["success"] > 0`, store the transition `K=10` additional times.

**Why this might work:** The fundamental problem is that SAC's uniform replay buffer treats a successful peg-insertion the same as a random flailing episode. After 300k transitions, the 500 steps from one successful episode are 0.17% of the buffer. With K=10 duplication, they become 1.7% — a 10× bias toward remembering success.

**Why NOT full SIL:** SIL replays past successful episodes as off-policy gradient updates with a separate loss. That's complex and changes SAC's optimization. We just want to bias the data distribution, not the algorithm. Simpler = more robust = easier to explain in a paper.

**Runs:** 2 tasks × 1-2 seeds = 2-4 runs

### Phase 3 — Full 5-seed runs (10–20 runs)

**Run ONLY if Phase 1 or Phase 2 shows clear advantage.**

**Decision logic:**

```
If demo_only or stabilized_demo > SAC on both tasks (last-20 mean):
  → Run 5 seeds × best method × 2 tasks = 10 runs
  → Run 5 seeds × SAC baseline × 2 tasks = 10 runs (if not done)
  → This is the paper's main table

If demo_only ≈ SAC (no clear winner):
  → Run 5 seeds × both × 2 tasks = 20 runs
  → Paper becomes "matches without task-specific engineering"

If demo_only < SAC on both:
  → STOP. Re-evaluate the approach entirely.
  → Consider sparse-reward setup (Section 6) or pivot to analysis paper
```

---

## 5. Paper Positioning (for each outcome)

### Outcome A: Stabilized demo > SAC (best case)

**Title:** "Self-Bootstrapped Demonstration Rewards with Experience Stabilization for Zero-Expert Manipulation"

**Core claim:** Without any expert data or task-specific reward engineering, the agent constructs its own demonstration buffer from early successes. With smoothed proximity reward and replay biasing, it converts initial discoveries into stable performance.

**Key differentiators:**
- vs TemporalOT (NeurIPS 2024): no expert demos needed
- vs SIL (NeurIPS 2018): feature-space proximity reward, not raw replay augmentation
- vs CAIMAN (arXiv 2025): grasping (prehensile), not pushing; self-improving buffer

**Required evidence:**
- 5-seed curves with error bars on 2+ tasks
- Ablation: demo_only (no stabilization) vs stabilized demo
- Comparison with SAC, SAC+HER
- PyBullet results as supplementary (prove the concept works on sparse reward)

### Outcome B: Demo ≈ SAC (decent case)

**Title:** "Zero-Expert Reward Shaping Matches Task-Specific Engineering in Manipulation RL"

**Framing:** The surprising result is NOT that we beat SAC — it's that we MATCH it without any task-specific knowledge. Meta-World's reward is hand-crafted. Ours is generated from the agent's own experience. Equivalent performance with zero domain knowledge is valuable.

**Combine with PyBullet:** On sparse/custom rewards (where no hand-crafted shaping exists), our method clearly outperforms SAC (72% → 92% success).

### Outcome C: Demo < SAC (worst case)

**Title:** "When and Why Self-Bootstrapped Rewards Fail on Dense Benchmarks"

**This is an analysis paper:**
- Show: strong results on sparse/custom rewards (PyBullet)
- Show: failure mode analysis on dense benchmarks (Meta-World)
- Insight: self-bootstrapped rewards help precisely when the base reward is insufficient
- Still publishable at ICLR/NeurIPS workshops

---

## 6. Optional Direction: Sparse-Reward Meta-World (high-risk, high-reward)

**Not in the immediate plan, but documented for consideration.**

A stronger experimental setup: instead of ADDING our reward to Meta-World's dense shaping, REPLACE it:

```python
r_total = sparse_success_indicator + basic_reach_reward + r_demo
```

Where:
- `sparse_success_indicator` = +1 if task complete, 0 otherwise
- `basic_reach_reward` = `-dist(hand, object)` (trivial, no domain knowledge)
- `r_demo` = self-bootstrapped proximity reward (our method)

**Why this is stronger:** It tests the REAL value proposition — can self-bootstrapped demos replace task-specific reward engineering? The comparison becomes:

| Method | Domain knowledge needed | Expected performance |
|---|---|---|
| SAC + Meta-World dense reward | Full task structure | Ceiling |
| SAC + sparse only | None | Floor |
| SAC + sparse + our demo reward | None | **If close to ceiling → paper** |
| TemporalOT + sparse + expert demos | Expert data required | Prior art benchmark |

**Risk:** Without dense reward, early exploration may never find success → empty demo buffer → stuck. Mitigation: the basic reach reward provides enough signal to discover contact by chance.

**When to try this:** Only if Phase 1-2 results are "demo ≈ SAC" on dense reward. This gives us a cleaner, more publishable experiment.

---

## 7. What We're Dropping

| Component | Status | Reason |
|---|---|---|
| CASID branding | Dropped | Causal is optional, not core. Misleading name. |
| Causal reward as primary contribution | Demoted to optional | Task-specific, redundant on dense benchmarks |
| True NN Granger (exp16) | Dead | Failed at 1M steps |
| Q-UCB potential shaping (exp17) | Dead | Marginal gain, complex |
| no_filter ablation | Complete | Showed causal filter hurts on peg-insert |
| causal_only ablation | Complete | Showed causal alone is task-specific |
| push-v3 experiments | Deprioritized | Push is understood; pick-place + peg-insert tell the story |
| CASID (full) as lead method | Replaced by demo_only | Simpler, cleaner, comparable performance |

---

## 8. Concrete Experiment Matrix

### Phase 1 (launching now): 12 runs

```
baseline_sac  pick-place-v3  seed=1   (GPU)
baseline_sac  pick-place-v3  seed=2   (GPU)
baseline_sac  pick-place-v3  seed=3   (GPU)
baseline_sac  peg-insert-v3  seed=1   (GPU)
baseline_sac  peg-insert-v3  seed=2   (GPU)
baseline_sac  peg-insert-v3  seed=3   (GPU)
demo_only     pick-place-v3  seed=1   (GPU)
demo_only     pick-place-v3  seed=2   (GPU)
demo_only     pick-place-v3  seed=3   (GPU)
demo_only     peg-insert-v3  seed=1   (GPU)
demo_only     peg-insert-v3  seed=2   (GPU)
demo_only     peg-insert-v3  seed=3   (GPU)
```



**Runtime:** ~18-24h (12 parallel, 1M steps each)

### Phase 2 (after Phase 1): 4-8 runs

Depends on Phase 1 results. Specifics defined above in Section 4.

---

## 9. Success Metric

**The paper needs this table to work:**

```
              | pick-place-v3 (5 seeds)  | peg-insert-v3 (5 seeds)
─────────────────────────────────────────────────────────────────
SAC           |  X.X ± Y.Y               |  X.X ± Y.Y
Demo (ours)   |  X.X ± Y.Y               |  X.X ± Y.Y
Demo+Stab     |  X.X ± Y.Y               |  X.X ± Y.Y
```

Where "Demo+Stab" ≥ SAC on at least one task, with the difference being statistically significant (p < 0.05 on a Welch t-test).

If we can't produce this table, we pivot to analysis (Outcome C in Section 5).
