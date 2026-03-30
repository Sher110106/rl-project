# Road to NeurIPS/ICML/ICLR: Research Plan
## "CASID: Causal Agency Self-Improving Dense Rewards for Robotic Manipulation"

**Written:** 2026-03-28
**Based on:** 19 experiments, cross-domain research, literature review
**Target:** NeurIPS 2026 (deadline ~May 2026) or ICML 2026 (deadline ~Feb 2026)

---

## Part 1: Honest Literature Gap Analysis

Before designing the paper, we need to know exactly where the boundary of existing work is, because reviewers will find every overlap.

### What already exists that overlaps with us

| Paper | Venue | What they do | Overlap with us | Key difference |
|---|---|---|---|---|
| **CAIMAN** (Yuan et al. 2025) arXiv:2502.00835 | arXiv (Mar 2026 v4) | Causal action influence as intrinsic reward for legged robot **pushing** | Our Granger correlation reward is conceptually identical | They: non-prehensile pushing, hierarchical control, learned dynamics model. We: prehensile grasping, flat policy, correlation proxy |
| **TemporalOT** (Fu et al. 2024) arXiv:2410.21795 | NeurIPS 2024 | OT distance to **expert video demos** as reward on Meta-World | Our Exp19 OT demo reward | They: require expert demos, temporal alignment. We: self-generated demos, no human input |
| **Jaques et al. 2019** ICLR | ICLR 2019 | Causal influence reward in **multi-agent** RL | Our causal agency measurement | Single-agent object manipulation is completely different setting |
| **SIL** (Oh et al. 2018) NeurIPS | NeurIPS 2018 | Replay past high-reward episodes as self-demonstrations | Our self-bootstrapped demo idea | SIL: direct experience replay augmentation. We: feature-space proximity reward shaping — different mechanism |
| **Empowerment-based manipulation** (Baldi et al. 2021) RSS | RSS 2021 | Mutual information intrinsic reward for sparse-reward manipulation | Our causal signal | They use full mutual information estimation (expensive). We use lightweight correlation. Different math formulation. |
| **HIntS** (2023) | — | Granger causality for **skill discovery** (offline, discrete) | Our Granger-causal reward | Offline skill discovery vs online dense reward — fundamentally different |

### The genuine gap

**No existing paper does all of the following simultaneously:**
1. Prehensile grasping (pick-up, not just pushing)
2. No human demonstrations required
3. Causal agency detection as a filtering criterion for self-generated demos
4. OT-inspired proximity reward to a growing, causally-filtered self-demo buffer
5. Systematic ablation showing each component's contribution

CAIMAN is the most dangerous overlap. But they do pushing with a legged robot and require a learned dynamics model. Our setting (grasping with arm, correlation proxy, self-demos) is sufficiently distinct — **but we must position very carefully in related work**.

The TemporalOT overlap is less dangerous because their key differentiation FROM us is temporal alignment of demos. Our key differentiation FROM them is **no human demos** — we bootstrap from our own successes.

---

## Part 2: The Proposed Method — CASID

### Core Insight

The reach→grasp→lift→place chain in manipulation RL has two failure modes:

**Local failure** (what to do when near the object): Agent reaches the cube but doesn't coordinate gripper closure. Standard distance rewards give no signal here.

**Global failure** (how to navigate state space): Agent doesn't know what "on track" looks like. No gradient toward successful-trajectory neighborhoods.

**CASID** solves both with two complementary signals that require zero human input:

```
CASID = r_env + α · r_causal(s,a) + β · r_demo(s)
```

where:
- `r_causal`: Causal agency reward — are your actions causally influencing the object?
- `r_demo`: Demo proximity reward — is your state near a state visited in a successful episode?

Both signals are **self-generated**: no human demonstrations, no expert data, no pre-training.

---

### Component 1: Causal Agency Reward

**Formal definition:**
Agent action sequence A_{t-k:t} causally influences object state O_{t:t+k} if:

```
Var(O_{t+1} | O_t, A_t) < Var(O_{t+1} | O_t)
```

i.e., knowing the action history reduces prediction error for object movement.

**Implementation (two versions, both to be reported):**

*Fast version (used in Exp5-14, correlation proxy):*
```python
r_causal_fast = max(0, corr(Δgripper[-4:], Δcube_pos[-4:])) * scale
```
Online, zero parameters, O(1) per step.

*Full version (NN Granger, Exp16/18):*
```python
# Causal model: MLP(gripper_hist[w] + cube_pos) → Δcube_pos
# Baseline model: MLP(cube_pos) → Δcube_pos
r_causal_nn = max(0, MSE(baseline) - MSE(causal)) * scale
```
Online SGD, ~500 parameters total.

**Finding from our experiments:** Fast version works (72%→100% with scale). NN version struggles within 1M steps (too slow to warm up). The paper should present the fast version as the practical method, and the NN version as the theoretically-grounded but harder-to-train alternative.

---

### Component 2: Self-Improving Demo Proximity Reward (the novel piece)

**The algorithm:**

```
Demo Buffer D = {} (empty at start)

Each episode:
  1. Run episode, collect trajectory τ = [(s₀,a₀), ..., (sT,aT)]
  2. If episode successful (r_total > threshold):
     2a. Compute causal_score(τ) = mean causal agency over episode
     2b. If causal_score > τ_causal:  # causal filter — only keep high-quality demos
         D ← D ∪ {features(sₜ) : t ∈ τ}  # add 6D feature states
     2c. Trim D to max size K (FIFO or quality-ranked)

Each step (when |D| > min_demos):
  3. r_demo(s) = max(0, σ - min_{d∈D} ||features(s) - d||₂) / σ * OT_SCALE
```

**The causal filter (novel):** Not all successful episodes are equal. A lucky episode where the cube rolled into the tray by chance gets no causal reward. Episodes where the agent actively grasped and lifted the cube get high causal reward. The filter ensures the demo buffer contains states from genuine manipulation, not noise.

**Why this is different from SIL:**
- SIL: replays successful episode *transitions* in the replay buffer (adds to experience)
- CASID: extracts *states* from successful episodes, uses them to shape the *reward function* at every step
- Different mechanism: replay augmentation vs reward shaping

**Why this is different from GAIL:**
- GAIL: requires expert demonstrations, adversarial training (unstable, expensive)
- CASID: self-generates its own demonstrations, no adversarial training, no expert data

**Why this is different from TemporalOT:**
- TemporalOT: aligns agent trajectory to expert video temporally
- CASID: no expert demos needed, no temporal alignment, proximity in feature space not trajectory space

---

### Unified CASID Reward

```
r_CASID = r_dense           (distance shaping — reach/tray)
        + r_gripper         (dense gripper proximity shaping)
        + α · r_causal      (causal agency signal)
        + β · r_demo        (self-improving demo proximity)
        + one-time bonuses  (contact, grasp, lift — anti-gaming)
```

Each component addresses a specific sub-task:
- `r_dense + r_gripper` → guides reaching and gripper approach
- `r_causal` → guides contact-phase coordination (when to close gripper)
- `r_demo` → guides global navigation toward successful-state neighborhoods

---

## Part 3: What the Paper Needs

### 3.1 The Non-Negotiables

**1. Standard benchmark environment (CRITICAL)**

Our custom PyBullet env is unpublishable at top venues. Reviewers cannot verify results on custom environments. We need one of:

**Option A: Meta-World** (recommended — NeurIPS/ICML gold standard for manipulation)
- 50 tasks with Sawyer 7-DOF arm in MuJoCo
- Standard baselines available (SAC, TD3+HER, GAIL, SIL)
- TemporalOT already tested here → we can directly compare
- Use subset: pick-place, push, peg-insert, drawer-open, button-press (5 representative tasks)

**Option B: RoboSuite + MuJoCo** (heavier but more realistic)
- Lift, PickPlace, Stack tasks
- More realistic physics, used in many CoRL/ICRA papers
- Harder to set up but more credible

**Option C: Gymnasium Robotics (FetchPickAndPlace)** (easiest port)
- Standard HER benchmark
- Already has SAC+HER baselines
- Less competitive than Meta-World but reviewers know it

**Recommendation: Meta-World.** It directly hosts TemporalOT's experiments, so we get a direct comparison for free.

**2. Multiple seeds (5 minimum)**

Every curve must have mean ± std. Single-seed results are desk-rejected.

**3. Proper baselines**

| Baseline | Why needed | Notes |
|---|---|---|
| SAC (no shaping) | Lower bound | Already have from our Exp2 essentially |
| SAC + HER | Standard manipulation baseline | Need to implement |
| SAC + SIL | Closest conceptual baseline | Need to implement (simple) |
| GAIL | Demo-based imitation baseline | Needs expert demos → use our successful episodes as "expert" |
| TemporalOT | OT-based demo reward | Their code is on GitHub (fuyw/TemporalOT) — run on same task |
| CASID (ours) | Our method | |
| CASID ablation: no causal filter | Ablation | Same as Exp19 without causal filter |
| CASID ablation: no demo reward | Ablation | Same as Exp14 (Granger only) |
| CASID ablation: no causal reward | Ablation | Demo proximity only, no Granger |

**4. Ablation studies**

Required ablations (minimum):
- A1: CASID vs CASID without causal filter (r_demo only) — isolates filter value
- A2: CASID vs CASID without r_demo (r_causal only) — isolates demo proximity value
- A3: CASID vs CASID with fixed demo buffer (no new demos added) — shows self-improvement
- A4: Causal scale sensitivity (0.2, 0.4, 0.8) — we already have this data
- A5: Demo buffer size sensitivity (100, 1k, 10k states)

**5. At least 3 tasks**

One task = workshop paper. Three tasks = conference paper. Five tasks = strong paper.

Recommended: pick-place, push, assembly (or peg-insert) — covers grasping, non-prehensile, precise placement.

---

### 3.2 The Theoretical Contribution

NeurIPS/ICML papers need theory. Options (pick one):

**Option A: Causal agency as manipulation progress measure (weaker but accessible)**
- Theorem: Under certain assumptions on the dynamics model, the causal agency signal is a consistent estimator of the causal influence of gripper actions on object state.
- Proof sketch: Connect to Granger causality definitions, show correlation proxy is consistent when signal-to-noise ratio is sufficient.

**Option B: CASID as approximate PBRS (stronger)**
- Theorem: The demo proximity reward r_demo approximates potential-based reward shaping Φ(s) = -min_{d∈D} ||features(s) - d||₂ as the demo buffer D converges to the successful-state distribution.
- Implication: As the agent improves, r_demo converges to a proper potential function — guaranteeing the optimal policy is preserved.
- This connects to Ng et al. 1999 reward shaping theory.

**Option C: Self-improving property (most novel)**
- Theorem: Under mild conditions, the expected causal quality of the demo buffer is non-decreasing in expectation (the agent's demos improve over time because better policies generate more causally-rich successful episodes).
- This is the "self-improving" formal claim.

**Recommendation: B + C together.** PBRS connection gives formal grounding; self-improving property is the novel theoretical insight.

---

### 3.3 Analysis and Understanding

Beyond numbers, reviewers want to understand **why** it works:

- **Demo buffer evolution plots**: Show how demo buffer grows and its causal quality improves over training
- **Causal reward activation heatmaps**: Show when r_causal is active (should peak during contact phase)
- **State trajectory visualization**: Show how demo proximity pulls agent trajectory toward successful neighborhoods
- **Sample efficiency curves**: Primary metric — how many steps to reach X% success
- **Task decomposition analysis**: Which sub-tasks (reach, grasp, lift, place) does each reward component help with?

---

## Part 4: Experiments to Run (Prioritized)

### Phase 1: Port to Meta-World (2-3 weeks)

**Action items:**
1. Install Meta-World: `pip install metaworld`
2. Implement CASID reward wrapper for Meta-World's `pick-place-v2` task
3. Match their eval protocol: 50 episodes per eval, success = task completion
4. Baseline SAC run to confirm env works

**Key tasks:**
- `pick-place-v2`: Our primary task (direct analog to current work)
- `push-v2`: Non-prehensile (tests generality beyond grasping)
- `assembly-v2` or `peg-insert-side-v2`: Precise placement (hardest)





| Method | Tasks | Seeds | Steps | Total runs |
|---|---|---|---|---|
| SAC (baseline) | pick-place, push, assembly | 5 each | 1M | 15 |
| SAC + HER | pick-place, push, assembly | 5 each | 1M | 15 |
| SAC + SIL | pick-place, push, assembly | 5 each | 1M | 15 |
| TemporalOT (their code) | pick-place, push, assembly | 5 each | 1M | 15 |
| CASID (ours) | pick-place, push, assembly | 5 each | 1M | 15 |
| CASID no filter | pick-place, push, assembly | 5 each | 1M | 15 |
| CASID no demo | pick-place, push, assembly | 5 each | 1M | 15 |
| CASID no causal | pick-place, push, assembly | 5 each | 1M | 15 |

**Total: 120 runs × ~5hrs each = 600 CPU-hours**
At 112 cores with OMP_NUM_THREADS=12 → run ~9 at a time → ~67 hours wall time (~3 days)

### Phase 3: Analysis runs (1-2 weeks)

- Demo buffer quality over time (log causal_score of incoming demos each episode)
- Ablation: demo buffer size (100, 1k, 10k states)
- Ablation: causal scale (0.2, 0.4, 0.8) — **we already have this from Wave 2!**
- 2M-step runs for pick-place to show continued improvement (we have exp8: 100% success)

---

## Part 5: Implementation Gaps to Fill

### 5.1 The Causal Filter (new code needed)

```python
class SelfImprovingDemoBuffer:
    def __init__(self, max_size=10_000, causal_threshold=0.1):
        self.states = []           # 6D feature states
        self.causal_scores = []    # per-episode mean causal reward
        self.max_size = max_size
        self.causal_threshold = causal_threshold
        self._tree = None          # BallTree, rebuilt when buffer changes

    def maybe_add_episode(self, episode_states, episode_causal_rewards, episode_success):
        if not episode_success:
            return
        causal_score = np.mean(episode_causal_rewards)
        if causal_score < self.causal_threshold:
            return  # Filter: not enough causal influence detected
        self.states.extend(episode_states)
        self.causal_scores.extend([causal_score] * len(episode_states))
        if len(self.states) > self.max_size:
            # Remove oldest states
            excess = len(self.states) - self.max_size
            self.states = self.states[excess:]
            self.causal_scores = self.causal_scores[excess:]
        self._tree = BallTree(np.array(self.states))  # rebuild index

    def proximity_reward(self, current_state, sigma=0.15, scale=0.3):
        if self._tree is None or len(self.states) < 50:
            return 0.0
        dist, _ = self._tree.query(current_state.reshape(1, -1), k=1)
        return max(0.0, sigma - dist[0, 0]) / sigma * scale
```

This is ~40 lines of code. The key new piece is the causal filter gate (`causal_score > threshold`).

### 5.2 Meta-World CASID wrapper (new code needed)

Port the env reward logic to work as a wrapper around Meta-World's `pick-place-v2`:
- Override `step()` to add CASID rewards on top of Meta-World's sparse success signal
- Feature extraction: same 6D (ee-object dist, object-goal dist, object_z, contact, grasp, gripper_state)
- Meta-World already provides these in its `info` dict

### 5.3 SIL baseline implementation

SIL is simple: maintain a secondary replay buffer of episodes with reward > threshold; sample 50% from this buffer during training. About 30 lines wrapping SB3's SAC replay buffer.

### 5.4 TemporalOT baseline

Their code is on GitHub (fuyw/TemporalOT). Clone and adapt to our benchmark tasks. Use our successful episodes as the "expert demo" (fair comparison — same data source as our method).

---

## Part 6: Paper Structure

### Title options
1. "CASID: Causal Agency Self-Improving Dense Rewards for Sample-Efficient Robotic Manipulation"
2. "Self-Bootstrapped Manipulation: Dense Rewards Without Human Demonstrations via Causal Agency Detection"
3. "Agency-Filtered Self-Imitation for Contact-Rich Robotic Manipulation"

### Abstract (draft)
*Learning robotic manipulation from scratch requires dense reward signals, yet hand-crafting rewards for contact-rich tasks is tedious and brittle. We present CASID, a reward engineering framework that bootstraps dense rewards from the agent's own experience without human demonstrations. CASID combines two complementary signals: (1) a causal agency reward that measures whether the agent's gripper actions causally influence object motion — directly rewarding active manipulation over passive proximity — and (2) a self-improving demo proximity reward that grows a buffer of successful-episode states and guides the agent toward those state neighborhoods. A causal filter ensures only genuinely manipulative episodes enter the demo buffer, preventing lucky failures from degrading signal quality. We prove that the demo proximity reward approximates a valid potential-based shaping function as the demo buffer converges, preserving policy optimality. Experiments on Meta-World across 3 manipulation tasks and 5 seeds show CASID achieves X% average success rate, outperforming SAC+HER, SIL, and TemporalOT (which requires human demonstrations) by Y%, Z%, and W% respectively, while reaching 90% success in 40% fewer steps.*

### Section outline

1. **Introduction** — manipulation bottleneck, dense reward problem, our solution
2. **Related Work** — causal influence (CAIMAN, Jaques), OT rewards (TemporalOT), SIL, empowerment — distinguish clearly from each
3. **Background** — Granger causality, PBRS theory (Ng 1999), OT distance
4. **Method** — CASID algorithm, both components, causal filter, theory
5. **Experiments** — Meta-World results, baselines, ablations
6. **Analysis** — demo buffer evolution, causal activation, sample efficiency breakdown
7. **Conclusion + Limitations** — single task family (manipulation), future: multi-object, deformable

---

## Part 7: Critical Risks and How to Mitigate

| Risk | Severity | Mitigation |
|---|---|---|
| CAIMAN (2025) is too close | HIGH | Emphasize prehensile grasping vs pushing; self-demo vs learned dynamics model; position as complementary |
| TemporalOT directly competes with Exp19 | HIGH | Our key claim: "no human demos needed". Their method needs expert videos. Frame CASID as the demo-free version. |
| True NN Granger (Exp16) fails at 1M steps | MEDIUM | Use fast correlation proxy as primary method; present NN Granger as a theoretically cleaner but slower variant |
| Meta-World port underperforms | MEDIUM | Start port early; adjust hyperparameters per task; the OT demo result on PyBullet (93%) suggests robustness |
| Single seed variance (seed=42 got 0%) | HIGH | This is why 5 seeds are mandatory; report mean ± std; acknowledge in limitations |
| SIL baseline outperforms us | MEDIUM | If SIL matches CASID, the paper's contribution weakens; the causal filter is our differentiator from SIL |
| Reviewers say "just use GAIL" | LOW | GAIL requires expert demos (adversarial training); we don't. Counter: we show CASID matches GAIL (with demos) while needing zero human input |

---

## Part 8: Timeline to Submission

**Target: NeurIPS 2026 (estimated deadline: May 15, 2026)**
Current date: March 28, 2026 — **7 weeks available**

| Week | Tasks |
|---|---|
| **Week 1** (Mar 28 – Apr 4) | Port to Meta-World; implement CASID wrapper; implement causal filter; verify SAC baseline runs |
| **Week 2** (Apr 4 – Apr 11) | Run SAC, SAC+HER, SIL baselines (5 seeds × 3 tasks = 45 runs) |
| **Week 3** (Apr 11 – Apr 18) | Run CASID + ablations (5 seeds × 3 tasks × 4 variants = 60 runs); TemporalOT baseline |
| **Week 4** (Apr 18 – Apr 25) | Analyze results; run analysis experiments (buffer evolution, heatmaps); write theory section |
| **Week 5** (Apr 25 – May 2) | Write paper (intro, method, experiments); prepare figures |
| **Week 6** (May 2 – May 9) | Polish, internal review, fix any failed experiments, rebuttal prep |
| **Week 7** (May 9 – May 15) | Final submission |

---

## Part 9: What We Already Have (Assets)

| Asset | Status | Value |
|---|---|---|
| PyBullet CASID implementation (Exp19 env) | Done | Proof of concept, 93% success |
| Causal correlation reward code | Done (Exp5-14) | Fast version of r_causal |
| True NN Granger code | Done (Exp16/18) | Full version, needs tuning |
| Q-UCB shaping code | Done (Exp17) | Potential for additional ablation |
| Demo extraction + BallTree lookup | Done (Exp19) | Core of r_demo |
| Scale ablation data (0.2, 0.4, 0.8) | Done (Exp13/5/14) | Ready for paper Figure |
| 2M-step result (Exp8: 100% success) | Done | Strong headline number |
| Seed variance data (5 seeds) | Done (Wave 2) | Shows robustness challenges |
| 10k-state demo buffer | Done (ot_demos.npy) | Ready to use |

**Biggest gap**: Meta-World port + running all baselines with proper seeds.

---

## Part 10: The Paper's Unique Claims (Summary)

What we claim that no existing paper can dispute:

1. **Self-bootstrapped grasping**: First method to achieve >90% pick-and-place success without any human demonstrations, using only the agent's own successful episodes as self-improving guidance.

2. **Causal filtering of self-generated demos**: Novel contribution — not all successful episodes are equal. Episodes where the agent actively grasped (high causal agency) produce better demos than lucky episodes. This filter improves demo quality over time.

3. **Complementary signals for manipulation sub-tasks**: Formal analysis showing causal agency reward addresses local (contact-phase) failures while demo proximity addresses global (navigation) failures — and together they are sufficient for contact-rich manipulation.

4. **No adversarial training, no human data, no pretrained models**: Simpler and more practical than GAIL, TemporalOT, or LfD approaches.

---

## Quick Reference: What to Build Next

**Status as of 2026-03-28:**

```
Priority 1 (this week):
  ✓ Install Meta-World, verify pick-place-v3 works (v3 is current, not v2)
  ✓ Implement CASID reward wrapper for Meta-World (casid_env.py)
  ✓ Implement SelfImprovingDemoBuffer with causal filter (in casid_env.py)
  ✓ Run CASID on pick-place-v3 seed 0 — RUNNING (exp20, early: 88.6 vs baseline 4.3)
  ✓ Run CASID on push-v3 seed 0 — RUNNING (exp22, early: 118 vs baseline push 20.2)
  ✓ Run CASID on peg-insert-side-v3 seed 0 — RUNNING (exp23, early: 18.2)
  ✓ Run Baseline SAC on pick-place-v3 seed 0 — RUNNING (exp21)
  ✓ Run Baseline SAC on push-v3 seed 0 — RUNNING (exp24)

Priority 2 (week 2):
  □ Verify seed 0 results (~18h)
  □ Run seeds 1-4: CASID + Baseline SAC (5 seeds × 3 tasks × 2 methods = 30 runs)
  □ Add SAC+HER baseline (5 seeds × 3 tasks)
  □ Add SIL baseline (5 seeds × 3 tasks)

Priority 3 (week 3):
  □ CASID ablations: no_filter, no_demo, no_causal (5 seeds × 3 tasks × 3 ablations)
  □ Clone and run TemporalOT on same tasks for direct comparison

Priority 4 (weeks 4-5):
  □ Write theory proofs (PBRS connection, self-improving property)
  □ Generate all figures
  □ Write paper
```

**Key experiment files:**
- `experiments/exp20_casid_metaworld/casid_env.py` — CASID wrapper + SelfImprovingDemoBuffer
- `experiments/exp20_casid_metaworld/train_casid.py` — CASID training (any Meta-World task)
- `experiments/exp21_baseline_metaworld/train_baseline.py` — Baseline SAC (any Meta-World task)
- Exp22/23/24 reuse the same scripts with different `--task` and `--logdir` flags
