"""
compare_results.py
-------------------
Loads diagnostics CSVs from all 3 experiments and generates
side-by-side comparison plots for the presentation.

Usage:
    python compare_results.py
    python compare_results.py --out results/comparison/

Outputs:
    comparison_dashboard.png  — 6-panel side-by-side
    comparison_reward.png     — reward overlay
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS = {
    "HER + SAC":       "exp1_her/logs/diagnostics/episode_diagnostics.csv",
    "Dense Gripper":   "exp2_dense/logs/diagnostics/episode_diagnostics.csv",
    "Curriculum":      "exp3_curriculum/logs/diagnostics/episode_diagnostics.csv",
}

COLORS = {
    "HER + SAC":     "#2196F3",
    "Dense Gripper": "#FF5722",
    "Curriculum":    "#4CAF50",
}


def load_experiment(csv_path):
    df = pd.read_csv(csv_path)
    # Handle both CSV formats (with/without stage column)
    agg = {
        "total_reward":  ("total_reward" if "total_reward" in df.columns
                          else "reward", "last" if "total_reward" in df.columns
                          else "sum"),
        "mean_ee_dist":  ("ee_cube_dist", "mean"),
        "mean_cube_z":   ("cube_z",       "mean"),
        "max_cube_z":    ("cube_z",       "max"),
        "contact_steps": ("any_contact" if "any_contact" in df.columns
                          else "contact" if "contact" in df.columns
                          else "any_contact", "sum"),
        "grasp_steps":   ("grasp",        "sum"),
        "steps":         ("step",         "max"),
    }
    # Filter to only columns that exist
    valid_agg = {}
    for key, (col, func) in agg.items():
        if col in df.columns:
            valid_agg[key] = (col, func)

    ep = df.groupby("episode").agg(**valid_agg).reset_index()
    return ep


def smooth(series, window=20):
    return series.rolling(window=window, min_periods=1).mean()


def plot_comparison(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ── Reward comparison overlay ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, ep in data.items():
        color = COLORS[name]
        ax.plot(ep["episode"], smooth(ep["total_reward"], 30),
                color=color, linewidth=2, label=name)
        ax.fill_between(ep["episode"],
                        smooth(ep["total_reward"], 50) - 20,
                        smooth(ep["total_reward"], 50) + 20,
                        alpha=0.1, color=color)
    ax.set_title("Reward Comparison Across Approaches", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_reward.png"), dpi=150)
    plt.close()
    print(f"Saved → comparison_reward.png")

    # ── 2×3 dashboard ──────────────────────────────────────────────
    metrics = [
        ("total_reward",  "Total Reward / Episode",     "Reward"),
        ("mean_ee_dist",  "Mean EE→Cube Dist (m)",      "Distance (m)"),
        ("mean_cube_z",   "Mean Cube Height (m)",        "Height (m)"),
        ("contact_steps", "Contact Steps / Episode",     "Steps"),
        ("grasp_steps",   "Grasp Steps / Episode",       "Steps"),
        ("steps",         "Episode Length",               "Steps"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experiment Comparison — HER vs Dense Gripper vs Curriculum",
                 fontsize=14, fontweight="bold")

    for idx, (col, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        for name, ep in data.items():
            if col in ep.columns:
                ax.plot(ep["episode"], smooth(ep[col], 30),
                        color=COLORS[name], linewidth=2, label=name)
        if col == "mean_cube_z":
            ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5,
                        label="lift threshold")
        if col == "steps":
            ax.axhline(500, color="red", linestyle="--", alpha=0.4,
                        label="max steps")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_dashboard.png"), dpi=150)
    plt.close()
    print(f"Saved → comparison_dashboard.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/comparison")
    args = parser.parse_args()

    data = {}
    for name, csv_path in EXPERIMENTS.items():
        if os.path.exists(csv_path):
            data[name] = load_experiment(csv_path)
            print(f"Loaded {name}: {len(data[name])} episodes")
        else:
            print(f"SKIP {name}: {csv_path} not found")

    if not data:
        print("No experiment data found. Run experiments first.")
        exit(1)

    plot_comparison(data, args.out)
