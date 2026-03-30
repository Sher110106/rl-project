"""
plot_results.py
---------------
Reads logs/diagnostics/episode_diagnostics.csv and produces 6 plots:

  1. Total reward per episode
  2. Mean EE→cube distance per episode
  3. Mean cube height (z) per episode
  4. Contact steps per episode
  5. Grasp steps per episode
  6. Episode length (detects early termination = success or failure)

Usage:
    python plot_results.py
    python plot_results.py --csv path/to/episode_diagnostics.csv
    python plot_results.py --out results/plots/

Outputs PNGs to results/plots/ (or --out directory).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")               # headless backend for remote servers
import matplotlib.pyplot as plt
import pandas as pd


def load(csv_path):
    df = pd.read_csv(csv_path)
    ep = df.groupby("episode").agg(
        total_reward   = ("total_reward",   "last"),
        mean_ee_dist   = ("ee_cube_dist",   "mean"),
        mean_cube_z    = ("cube_z",         "mean"),
        max_cube_z     = ("cube_z",         "max"),
        contact_steps  = ("any_contact",    "sum"),
        grasp_steps    = ("grasp",          "sum"),
        steps          = ("step",           "max"),
    ).reset_index()
    return ep


def smooth(series, window=20):
    return series.rolling(window=window, min_periods=1).mean()


def plot(ep, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    episodes = ep["episode"]

    # ── Combined 2×3 dashboard ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("SAC Training — Pick-and-Place Diagnostics (v4, anti-gaming)",
                 fontsize=14, fontweight="bold")

    panels = [
        (axes[0, 0], "total_reward",  "steelblue",  "Total Reward / Episode",          "Reward"),
        (axes[0, 1], "mean_ee_dist",  "tomato",     "Mean EE→Cube Dist / Episode (m)", "Distance (m)"),
        (axes[0, 2], "mean_cube_z",   "seagreen",   "Mean Cube Height / Episode (m)",  "Height (m)"),
        (axes[1, 0], "contact_steps", "darkorange", "Contact Steps / Episode",         "Steps"),
        (axes[1, 1], "grasp_steps",   "purple",     "Grasp Steps / Episode",           "Steps"),
        (axes[1, 2], "steps",         "grey",       "Episode Length",                  "Steps"),
    ]

    for ax, col, color, title, ylabel in panels:
        ax.plot(episodes, ep[col], alpha=0.2, color=color)
        ax.plot(episodes, smooth(ep[col]), color=color, linewidth=2)
        if col == "mean_cube_z":
            ax.axhline(0.05, color=color, linestyle="--", alpha=0.5,
                        label="lift threshold")
            ax.legend(fontsize=8)
        if col == "steps":
            ax.axhline(500, color="red", linestyle="--", alpha=0.4,
                        label="max steps")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_diagnostics.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close()

    # ── Individual slide-ready PNGs ─────────────────────────────────────
    individual = {
        "reward":   ("total_reward",  "steelblue",  "Total Reward / Episode",              "Reward"),
        "ee_dist":  ("mean_ee_dist",  "tomato",     "Mean EE→Cube Distance / Episode (m)", "Distance (m)"),
        "cube_z":   ("mean_cube_z",   "seagreen",   "Mean Cube Height / Episode (m)",      "Height (m)"),
        "contacts": ("contact_steps", "darkorange", "Contact Steps / Episode",             "Steps"),
        "grasps":   ("grasp_steps",   "purple",     "Grasp Steps / Episode",               "Steps"),
        "ep_len":   ("steps",         "grey",       "Episode Length (shorter = success)",   "Steps"),
    }
    for name, (col, color, title, ylabel) in individual.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(episodes, ep[col], alpha=0.2, color=color)
        ax.plot(episodes, smooth(ep[col]), color=color, linewidth=2)
        if col == "mean_cube_z":
            ax.axhline(0.05, color=color, linestyle="--", alpha=0.5,
                        label="lift threshold")
            ax.legend()
        if col == "steps":
            ax.axhline(500, color="red", linestyle="--", alpha=0.4,
                        label="max steps")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/diagnostics/episode_diagnostics.csv")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Run training first: python train.py --algo sac")
        exit(1)

    ep = load(args.csv)
    print(f"Loaded {len(ep)} episodes from {args.csv}")
    plot(ep, args.out)
