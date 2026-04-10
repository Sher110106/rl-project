#!/usr/bin/env python3
"""
generate_figures.py
===================
Generates all 4 paper figures for the SAC entropy bifurcation paper.

Outputs:
  paper/figures/figure1_bifurcation.pdf
  paper/figures/figure2_failure_modes.pdf
  paper/figures/figure3_ablation.pdf
  paper/figures/figure4_mechanistic.pdf

Run from: /Users/sher/project/rlp/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# PMLR style settings
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
})

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP36_LOGS = os.path.join(BASE, "experiments/exp36_alpha_trajectory/logs")
EXP35_LOGS = os.path.join(BASE, "experiments/exp35_causal_ablation/logs")
EXP35_ANALYSIS = os.path.join(BASE, "experiments/exp35_causal_ablation/analysis")
FIGURES_DIR = os.path.join(BASE, "paper/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Color palette (colorblind-friendly)
# ─────────────────────────────────────────────────────────────────────────────
C_SOLVED   = '#D95F02'   # warm orange — solved seeds
C_COLLAPSE = '#2C3E50'   # dark navy — α-collapse
C_EXPLODE  = '#8E44AD'   # purple — α-explosion
C_FORGET   = '#7F8C8D'   # gray — discover-then-forget
C_A        = '#D95F02'   # Method A: SAC auto
C_B        = '#377EB8'   # Method B: anneal
C_C        = '#E6AB02'   # Method C: demo
C_D        = '#1B9E77'   # Method D: demo+anneal
ALPHA_ZONE_LOW  = 0.01
ALPHA_ZONE_HIGH = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_alpha_trajectory(task, seed):
    path = os.path.join(EXP36_LOGS, f"{task}__seed{seed}/alpha_trajectory.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df['step'].values, df['alpha'].values


def load_eval_rewards(task, seed, source='exp36'):
    if source == 'exp36':
        path = os.path.join(EXP36_LOGS, f"{task}__seed{seed}/eval/evaluations.npz")
    else:
        path = os.path.join(EXP35_LOGS, f"{task}__{seed}/eval/evaluations.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    steps = data['timesteps']
    rewards = data['results'].mean(axis=1)
    return steps, rewards


def load_exp35_eval(task, method, seed):
    run_id = f"{task}__method{method}__seed{seed}"
    path = os.path.join(EXP35_LOGS, f"{run_id}/eval/evaluations.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    steps = data['timesteps']
    rewards = data['results'].mean(axis=1)
    return steps, rewards


# ─────────────────────────────────────────────────────────────────────────────
# Seed classification for peg-insert and pick-place (exp36, 12 seeds)
# ─────────────────────────────────────────────────────────────────────────────
PEG_FAILED  = {4: 'collapse'}
PICK_FAILED = {1: 'explode', 3: 'collapse', 6: 'collapse'}
PICK_FORGET = {4}   # discover-then-forget — still solved but worth noting

def seed_color(task, seed):
    if task == 'peg-insert-side-v3':
        fm = PEG_FAILED.get(seed)
    else:
        fm = PICK_FAILED.get(seed)
    if fm == 'collapse':  return C_COLLAPSE
    if fm == 'explode':   return C_EXPLODE
    if seed in PICK_FORGET: return C_FORGET
    return C_SOLVED

def seed_style(task, seed):
    if task == 'peg-insert-side-v3':
        if seed in PEG_FAILED: return '--', 1.5
    else:
        if seed in PICK_FAILED: return '--', 1.5
    return '-', 0.8

def seed_zorder(task, seed):
    if task == 'peg-insert-side-v3':
        return 3 if seed in PEG_FAILED else 2
    return 3 if seed in PICK_FAILED else 2


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — The α Bifurcation (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────
def make_figure1():
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.0))
    tasks = ['peg-insert-side-v3', 'pick-place-v3']
    titles_top = ['peg-insert-side-v3 (11/12 solved)', 'pick-place-v3 (9/12 solved)']

    for col, task in enumerate(tasks):
        ax_alpha  = axes[0, col]
        ax_reward = axes[1, col]

        for seed in range(12):
            steps, alpha = load_alpha_trajectory(task, seed)
            if steps is None:
                continue
            color = seed_color(task, seed)
            ls, lw = seed_style(task, seed)
            zo = seed_zorder(task, seed)
            alpha_val = 0.95 if color != C_SOLVED else 0.55
            ax_alpha.plot(steps / 1e6, alpha, color=color, ls=ls,
                         lw=lw, alpha=alpha_val, zorder=zo)

            # Reward
            rsteps, rewards = load_eval_rewards(task, seed)
            if rsteps is not None:
                ax_reward.plot(rsteps / 1e6, rewards, color=color,
                              ls=ls, lw=lw, alpha=alpha_val, zorder=zo)

        # Alpha stable zone band
        ax_alpha.axhspan(ALPHA_ZONE_LOW, ALPHA_ZONE_HIGH, alpha=0.08,
                        color='green', label='Stable zone')
        ax_alpha.set_yscale('log')
        ax_alpha.set_ylim(3e-4, 30)
        ax_alpha.set_title(titles_top[col], fontsize=8, pad=3)
        ax_alpha.set_xlabel('')
        ax_alpha.set_ylabel(r'$\alpha$ (log scale)' if col == 0 else '', usetex=False)
        ax_alpha.grid(True, which='both', ls='--')
        ax_alpha.set_xlim(0, 1.05)
        if col == 1:
            ax_alpha.set_yticklabels([])

        ax_reward.set_xlabel('Training steps ($\\times 10^6$)')
        ax_reward.set_ylabel('Mean eval reward' if col == 0 else '')
        ax_reward.axhline(500, color='k', ls=':', lw=0.8, alpha=0.5)
        ax_reward.set_xlim(0, 1.0)
        ax_reward.grid(True, ls='--')
        if col == 1:
            ax_reward.set_yticklabels([])

    # Legend
    legend_elements = [
        Line2D([0], [0], color=C_SOLVED,   lw=1.2, label='Solved'),
        Line2D([0], [0], color=C_COLLAPSE, lw=1.5, ls='--', label=r'$\alpha$-collapse'),
        Line2D([0], [0], color=C_EXPLODE,  lw=1.5, ls='--', label=r'$\alpha$-explosion'),
        Line2D([0], [0], color='green',    lw=4, alpha=0.25, label='Stable zone [0.01, 0.30]'),
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=6.5,
                      framealpha=0.85)

    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.5)
    out = os.path.join(FIGURES_DIR, 'figure1_bifurcation.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Three Failure Modes (1×3 panels, dual y-axis each)
# ─────────────────────────────────────────────────────────────────────────────
def make_figure2():
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.3))

    # Panel configs: (task, seed, title, alpha_color, failure_label)
    panels = [
        ('pick-place-v3',      3, r'(a) $\alpha$-collapse',      C_COLLAPSE, 'collapse'),
        ('pick-place-v3',      1, r'(b) $\alpha$-explosion',     C_EXPLODE,  'explode'),
        ('pick-place-v3',      4, r'(c) Discover-then-forget',   C_FORGET,   'forget'),
    ]

    for ax, (task, seed, title, color, mode) in zip(axes, panels):
        steps_a, alpha = load_alpha_trajectory(task, seed)
        steps_r, rewards = load_eval_rewards(task, seed)

        ax2 = ax.twinx()

        if steps_a is not None:
            ax.plot(steps_a / 1e6, alpha, color=color, lw=1.5, label=r'$\alpha$')
        if steps_r is not None:
            ax2.plot(steps_r / 1e6, rewards, color='#666666', lw=1.0,
                    ls='--', alpha=0.8, label='Reward')

        ax.set_yscale('log')
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel('Steps ($\\times 10^6$)', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax2.tick_params(axis='both', labelsize=7)
        ax.set_xlim(0, 1.0)

        if axes.tolist().index(ax) == 0:
            ax.set_ylabel(r'$\alpha$ (log scale)', fontsize=8)
        else:
            ax.set_yticklabels([])

        if axes.tolist().index(ax) == 2:
            ax2.set_ylabel('Eval reward', fontsize=8)
        else:
            ax2.set_yticklabels([])

        ax.grid(True, which='both', ls='--', alpha=0.4)

        # Small legend
        lines1 = [Line2D([0],[0], color=color, lw=1.5, label=r'$\alpha$')]
        lines2 = [Line2D([0],[0], color='#666666', lw=1.0, ls='--', label='Reward')]
        ax.legend(handles=lines1+lines2, fontsize=6, loc='upper right')

    plt.tight_layout(pad=0.4, w_pad=0.8)
    out = os.path.join(FIGURES_DIR, 'figure2_failure_modes.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Ablation Learning Curves (exp35, 2 panels)
# ─────────────────────────────────────────────────────────────────────────────
def make_figure3():
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
    tasks = ['peg-insert-side-v3', 'pick-place-v3']
    task_labels = ['peg-insert-side-v3', 'pick-place-v3']
    methods = ['A', 'B', 'C', 'D']
    method_colors = [C_A, C_B, C_C, C_D]
    method_labels = ['A: SAC (auto-α)', 'B: SAC (anneal)', 'C: demo (auto-α)', 'D: demo (anneal)']
    n_seeds = 8

    for ax, task, task_label in zip(axes, tasks, task_labels):
        for method, color, label in zip(methods, method_colors, method_labels):
            all_rewards = []
            all_steps = None
            for seed in range(n_seeds):
                steps, rewards = load_exp35_eval(task, method, seed)
                if steps is not None and rewards is not None:
                    if all_steps is None:
                        all_steps = steps
                    # Align to common length
                    min_len = min(len(all_steps), len(rewards))
                    all_steps = all_steps[:min_len]
                    all_rewards.append(rewards[:min_len])

            if not all_rewards:
                continue

            min_len = min(len(r) for r in all_rewards)
            arr = np.array([r[:min_len] for r in all_rewards])
            all_steps = all_steps[:min_len]
            mean_r = arr.mean(axis=0)
            std_r  = arr.std(axis=0)

            ax.plot(all_steps / 1e6, mean_r, color=color, lw=1.2, label=label)
            ax.fill_between(all_steps / 1e6, mean_r - std_r, mean_r + std_r,
                           color=color, alpha=0.15)

        ax.axhline(500, color='k', ls=':', lw=0.8, alpha=0.6, label='Success threshold')
        ax.set_title(task_label, fontsize=8, pad=3)
        ax.set_xlabel('Training steps ($\\times 10^6$)', fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls='--', alpha=0.4)
        ax.tick_params(labelsize=7)
        if axes.tolist().index(ax) == 0:
            ax.set_ylabel('Mean eval reward', fontsize=8)
        else:
            ax.set_yticklabels([])

    axes[1].legend(fontsize=6.5, loc='upper left', framealpha=0.9)
    plt.tight_layout(pad=0.4, w_pad=0.6)
    out = os.path.join(FIGURES_DIR, 'figure3_ablation.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Mechanistic: Buffer success + Q-value probes (exp35)
# ─────────────────────────────────────────────────────────────────────────────
def load_buffer_success(task, method, seed):
    run_id = f"{task}__method{method}__seed{seed}"
    path = os.path.join(EXP35_LOGS, f"{run_id}/buffer_success_log.csv")
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
        if df.empty or 'step' not in df.columns:
            return None, None
        return df['step'].values, df['buffer_success_fraction'].values
    except Exception:
        return None, None


def load_qvalue_probe(task, method, seed):
    run_id = f"{task}__method{method}__seed{seed}"
    path = os.path.join(EXP35_LOGS, f"{run_id}/qvalue_probe_log.csv")
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
        if df.empty or 'mean_q' not in df.columns:
            return None, None
        return df['step'].values, df['mean_q'].values
    except Exception:
        return None, None


def make_figure4():
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))

    methods = ['A', 'B', 'C', 'D']
    method_colors = [C_A, C_B, C_C, C_D]
    method_labels = ['A: SAC (auto-α)', 'B: SAC (anneal)', 'C: demo (auto-α)', 'D: demo (anneal)']
    n_seeds = 8

    # Left: buffer success fraction — peg-insert
    ax = axes[0]
    task = 'peg-insert-side-v3'
    for method, color, label in zip(methods, method_colors, method_labels):
        all_fracs = []
        all_steps = None
        for seed in range(n_seeds):
            steps, frac = load_buffer_success(task, method, seed)
            if steps is not None:
                if all_steps is None:
                    all_steps = steps
                min_len = min(len(all_steps), len(frac))
                all_steps = all_steps[:min_len]
                all_fracs.append(frac[:min_len])
        if all_fracs:
            min_len = min(len(f) for f in all_fracs)
            arr = np.array([f[:min_len] for f in all_fracs])
            all_steps = all_steps[:min_len]
            mean_f = arr.mean(axis=0)
            std_f  = arr.std(axis=0)
            ax.plot(all_steps / 1e6, mean_f, color=color, lw=1.2, label=label)
            ax.fill_between(all_steps / 1e6, mean_f - std_f, mean_f + std_f,
                           color=color, alpha=0.15)

    ax.set_title('Replay buffer success fraction\n(peg-insert-side-v3)', fontsize=8, pad=3)
    ax.set_xlabel('Training steps (×10⁶)', fontsize=8)
    ax.set_ylabel('Success fraction (reward ≥ 500)', fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, ls='--', alpha=0.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc='upper left')

    # Right: Q-value probes — pick-place
    ax = axes[1]
    task = 'pick-place-v3'
    for method, color, label in zip(methods, method_colors, method_labels):
        for seed in range(n_seeds):
            steps, qvals = load_qvalue_probe(task, method, seed)
            if steps is None or qvals is None:
                continue
            # Plot individual seeds for annealed methods to show divergence
            if method in ['B', 'D']:
                # Clip extreme values for display
                qvals_disp = np.clip(qvals, -100, 20000)
                ax.plot(steps / 1e6, qvals_disp, color=color, lw=0.7, alpha=0.6)
            else:
                ax.plot(steps / 1e6, qvals, color=color, lw=0.7, alpha=0.5)

    # Add method legend via dummy lines
    legend_els = [Line2D([0],[0], color=c, lw=1.2, label=l)
                  for c, l in zip(method_colors, method_labels)]
    ax.set_title('Q-values at probe states\n(pick-place-v3)', fontsize=8, pad=3)
    ax.set_xlabel('Training steps (×10⁶)', fontsize=8)
    ax.set_ylabel('Mean Q(s, a)', fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-50, 1000)
    ax.grid(True, ls='--', alpha=0.4)
    ax.tick_params(labelsize=7)
    ax.legend(handles=legend_els, fontsize=6.5, loc='upper left')
    ax.text(0.97, 0.95, 'B,D seeds spike to\n12k–17k (clipped)',
            transform=ax.transAxes, fontsize=6, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    plt.tight_layout(pad=0.4, w_pad=0.6)
    out = os.path.join(FIGURES_DIR, 'figure4_mechanistic.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures...")
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    print("All figures generated.")
