"""Collect and summarize all final 5-seed results for the paper table."""
import numpy as np
import os

BASE = "/Users/sher/project/rlp/results"

# Map: (method, task, seed) -> path to evaluations.npz
PATHS = {
    # SAC baseline
    ("sac", "pick-place-v3", 0): f"{BASE}/experiments/metaworld/exp21_baseline_metaworld/logs/pick-place-v3_seed0/eval/evaluations.npz",
    ("sac", "pick-place-v3", 1): f"{BASE}/final_runs/phase2_baseline/logs/pick-place-v3_seed1/eval/evaluations.npz",
    ("sac", "pick-place-v3", 2): f"{BASE}/final_runs/phase2_baseline/logs/pick-place-v3_seed2/eval/evaluations.npz",
    ("sac", "pick-place-v3", 3): f"{BASE}/final_runs/phase2_baseline/logs/pick-place-v3_seed3/eval/evaluations.npz",
    ("sac", "pick-place-v3", 4): f"{BASE}/final_runs/final_sac/logs/pick-place-v3_seed4/eval/evaluations.npz",

    ("sac", "peg-insert-side-v3", 0): f"{BASE}/experiments/metaworld/exp34_baseline_peg/logs/peg-insert-side-v3_seed0/eval/evaluations.npz",
    ("sac", "peg-insert-side-v3", 1): f"{BASE}/final_runs/phase2_baseline/logs/peg-insert-side-v3_seed1/eval/evaluations.npz",
    ("sac", "peg-insert-side-v3", 2): f"{BASE}/final_runs/phase2_baseline/logs/peg-insert-side-v3_seed2/eval/evaluations.npz",
    ("sac", "peg-insert-side-v3", 3): f"{BASE}/final_runs/phase2_baseline/logs/peg-insert-side-v3_seed3/eval/evaluations.npz",
    ("sac", "peg-insert-side-v3", 4): f"{BASE}/final_runs/final_sac/logs/peg-insert-side-v3_seed4/eval/evaluations.npz",

    # demo_smooth
    ("demo_smooth", "pick-place-v3", 0): f"{BASE}/experiments/metaworld/phase2_demo_smooth/logs/pick-place-v3_seed0/eval/evaluations.npz",
    ("demo_smooth", "pick-place-v3", 1): f"{BASE}/final_runs/final_demo_smooth/logs/pick-place-v3_seed1/eval/evaluations.npz",
    ("demo_smooth", "pick-place-v3", 2): f"{BASE}/final_runs/final_demo_smooth/logs/pick-place-v3_seed2/eval/evaluations.npz",
    ("demo_smooth", "pick-place-v3", 3): f"{BASE}/final_runs/final_demo_smooth/logs/pick-place-v3_seed3/eval/evaluations.npz",
    ("demo_smooth", "pick-place-v3", 4): f"{BASE}/final_runs/final_demo_smooth/logs/pick-place-v3_seed4/eval/evaluations.npz",

    ("demo_smooth", "peg-insert-side-v3", 0): f"{BASE}/experiments/metaworld/phase2_demo_smooth/logs/peg-insert-side-v3_seed0/eval/evaluations.npz",
    ("demo_smooth", "peg-insert-side-v3", 1): f"{BASE}/final_runs/final_demo_smooth/logs/peg-insert-side-v3_seed1/eval/evaluations.npz",
    ("demo_smooth", "peg-insert-side-v3", 2): f"{BASE}/final_runs/final_demo_smooth/logs/peg-insert-side-v3_seed2/eval/evaluations.npz",
    ("demo_smooth", "peg-insert-side-v3", 3): f"{BASE}/final_runs/final_demo_smooth/logs/peg-insert-side-v3_seed3/eval/evaluations.npz",
    ("demo_smooth", "peg-insert-side-v3", 4): f"{BASE}/final_runs/final_demo_smooth/logs/peg-insert-side-v3_seed4/eval/evaluations.npz",

    # smooth+anneal
    ("smooth_anneal", "pick-place-v3", 0): f"{BASE}/final_runs/final_anneal/logs/pick-place-v3_seed0/eval/evaluations.npz",
    ("smooth_anneal", "pick-place-v3", 1): f"{BASE}/final_runs/final_anneal/logs/pick-place-v3_seed1/eval/evaluations.npz",
    ("smooth_anneal", "pick-place-v3", 2): f"{BASE}/final_runs/final_anneal/logs/pick-place-v3_seed2/eval/evaluations.npz",
    ("smooth_anneal", "pick-place-v3", 3): f"{BASE}/final_runs/final_anneal/logs/pick-place-v3_seed3/eval/evaluations.npz",
    ("smooth_anneal", "pick-place-v3", 4): f"{BASE}/final_runs/final_anneal/logs/pick-place-v3_seed4/eval/evaluations.npz",

    ("smooth_anneal", "peg-insert-side-v3", 0): f"{BASE}/experiments/metaworld/phase2_smooth_anneal/logs/peg-insert-side-v3_seed0/eval/evaluations.npz",
    ("smooth_anneal", "peg-insert-side-v3", 1): f"{BASE}/experiments/metaworld/phase2_smooth_anneal/logs/peg-insert-side-v3_seed1/eval/evaluations.npz",
    ("smooth_anneal", "peg-insert-side-v3", 2): f"{BASE}/experiments/metaworld/phase2_smooth_anneal/logs/peg-insert-side-v3_seed2/eval/evaluations.npz",
    ("smooth_anneal", "peg-insert-side-v3", 3): f"{BASE}/final_runs/final_anneal/logs/peg-insert-side-v3_seed3/eval/evaluations.npz",
    ("smooth_anneal", "peg-insert-side-v3", 4): f"{BASE}/final_runs/final_anneal/logs/peg-insert-side-v3_seed4/eval/evaluations.npz",
}

def load_returns(path):
    """Load mean episodic returns per eval checkpoint."""
    d = np.load(path)
    # SB3 evaluations.npz: 'results' shape (n_evals, n_eval_eps), 'timesteps'
    results = d['results']  # shape: (n_evals, n_eval_eps)
    timesteps = d['timesteps']
    mean_per_eval = results.mean(axis=1)
    return timesteps, mean_per_eval

def last_n_mean(returns, n=20):
    return float(np.mean(returns[-n:]))

def peak(returns):
    return float(np.max(returns))

def time_to_first_success(timesteps, returns, threshold=500):
    """First timestep where eval mean > threshold."""
    for t, r in zip(timesteps, returns):
        if r > threshold:
            return int(t)
    return None

# Collect per-seed data
data = {}
for (method, task, seed), path in PATHS.items():
    if not os.path.exists(path):
        print(f"MISSING: {method} {task} seed{seed}: {path}")
        continue
    ts, rets = load_returns(path)
    data[(method, task, seed)] = {
        "timesteps": ts,
        "returns": rets,
        "last20": last_n_mean(rets, 20),
        "peak": peak(rets),
        "tts": time_to_first_success(ts, rets, threshold=500),
    }

# Summarize
METHODS = ["sac", "demo_smooth", "smooth_anneal"]
TASKS = ["pick-place-v3", "peg-insert-side-v3"]
METHOD_LABELS = {"sac": "SAC (baseline)", "demo_smooth": "demo_smooth (k=5)", "smooth_anneal": "smooth+anneal"}
TASK_LABELS = {"pick-place-v3": "pick-place", "peg-insert-side-v3": "peg-insert"}

print("\n" + "="*80)
print("PAPER TABLE: 5-Seed Results on Meta-World")
print("="*80)

for task in TASKS:
    print(f"\nTask: {TASK_LABELS[task]}")
    print(f"{'Method':<25} {'Last-20 Mean±Std':>20} {'Peak Mean±Std':>20} {'TTS (mean steps)':>18}")
    print("-"*85)
    for method in METHODS:
        seeds = [s for s in range(5) if (method, task, s) in data]
        if not seeds:
            print(f"{METHOD_LABELS[method]:<25} {'N/A':>20}")
            continue
        last20s = [data[(method, task, s)]["last20"] for s in seeds]
        peaks = [data[(method, task, s)]["peak"] for s in seeds]
        ttss = [data[(method, task, s)]["tts"] for s in seeds if data[(method, task, s)]["tts"] is not None]

        l20_mean = np.mean(last20s)
        l20_std = np.std(last20s)
        pk_mean = np.mean(peaks)
        pk_std = np.std(peaks)
        tts_str = f"{np.mean(ttss)/1000:.0f}k" if ttss else "never"

        n_seeds = len(seeds)
        seed_str = f"(n={n_seeds})" if n_seeds < 5 else ""
        print(f"{METHOD_LABELS[method]:<25} {f'{l20_mean:.0f} ± {l20_std:.0f}':>20} {f'{pk_mean:.0f} ± {pk_std:.0f}':>20} {tts_str:>18}  {seed_str}")

# Per-seed breakdown
print("\n" + "="*80)
print("PER-SEED DETAIL")
print("="*80)
for task in TASKS:
    print(f"\nTask: {TASK_LABELS[task]}")
    for method in METHODS:
        print(f"  {METHOD_LABELS[method]}:")
        for s in range(5):
            if (method, task, s) in data:
                d = data[(method, task, s)]
                tts = f"{d['tts']//1000}k" if d['tts'] else "never"
                print(f"    seed{s}: last20={d['last20']:.0f}  peak={d['peak']:.0f}  tts={tts}")
            else:
                print(f"    seed{s}: MISSING")

# Save summary for plot script
import json
summary = {}
for (method, task, seed), d in data.items():
    key = f"{method}__{task}__{seed}"
    summary[key] = {
        "last20": d["last20"],
        "peak": d["peak"],
        "tts": d["tts"],
        "timesteps": d["timesteps"].tolist(),
        "returns": d["returns"].tolist(),
    }
out_path = "/Users/sher/project/rlp/results/final_summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[Saved summary to {out_path}]")
