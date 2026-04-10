#!/usr/bin/env python3
"""
exp35_mechanistic_analysis.py
=============================
Mine all mechanistic logs from the 64-run causal ablation experiment.

Key question: What does SAC's auto-tuned α actually do on seeds that solve
vs seeds that don't? Does SAC naturally anneal on success?

Usage:
    python exp35_mechanistic_analysis.py --base_dir experiments/exp35_causal_ablation/logs

Outputs 8 analysis plots + 1 summary CSV to experiments/exp35_causal_ablation/analysis/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TASKS = ['peg-insert-side-v3', 'pick-place-v3']
METHODS = {
    'A': 'SAC baseline',
    'B': 'SAC + anneal',
    'C': 'demo_smooth',
    'D': 'demo_smooth + anneal',
}
SEEDS = list(range(8))
SUCCESS_THRESHOLD = 500  # episode reward >= this = success

# Matplotlib style
plt.rcParams.update({
    'figure.figsize': (14, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

METHOD_COLORS = {'A': '#E24B4A', 'B': '#378ADD', 'C': '#BA7517', 'D': '#1D9E75'}
METHOD_LABELS = {k: v for k, v in METHODS.items()}


def find_run_dir(base_dir, task, method, seed):
    """Find the directory for a specific run. Adapt pattern to your naming."""
    # Try common patterns
    patterns = [
        f"{task}__method{method}__seed{seed}",
        f"{task.replace('-', '_')}__method{method}__seed{seed}",
        f"{task}__method_{method}__seed_{seed}",
    ]
    for p in patterns:
        path = os.path.join(base_dir, p)
        if os.path.isdir(path):
            return path
    # Fallback: glob
    import glob
    matches = glob.glob(os.path.join(base_dir, f"*{task}*method*{method}*seed*{seed}*"))
    if matches:
        return matches[0]
    return None


def load_eval_data(run_dir):
    """Load evaluation results from evaluations.npz"""
    npz_path = os.path.join(run_dir, 'eval', 'evaluations.npz')
    if not os.path.exists(npz_path):
        return None, None
    data = np.load(npz_path)
    timesteps = data['timesteps']
    results = data['results']  # shape: (n_checkpoints, n_eval_episodes)
    return timesteps, results


def load_csv_safe(path, **kwargs):
    """Load a CSV, return None if missing."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# ANALYSIS 1: Learning curves (mean ± std across seeds)
# ─────────────────────────────────────────────────────────────
def plot_learning_curves(base_dir, out_dir):
    """Plot reward vs steps for all 4 methods per task."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        for method_id in ['A', 'B', 'C', 'D']:
            all_curves = []
            for seed in SEEDS:
                run_dir = find_run_dir(base_dir, task, method_id, seed)
                if run_dir is None:
                    continue
                ts, results = load_eval_data(run_dir)
                if ts is None:
                    continue
                mean_rewards = results.mean(axis=1)
                all_curves.append(mean_rewards)
            
            if not all_curves:
                continue
            
            # Align to shortest
            min_len = min(len(c) for c in all_curves)
            curves = np.array([c[:min_len] for c in all_curves])
            timesteps = ts[:min_len]
            
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            
            ax.plot(timesteps / 1000, mean, color=METHOD_COLORS[method_id],
                    label=f"{method_id}: {METHOD_LABELS[method_id]}", linewidth=1.5)
            ax.fill_between(timesteps / 1000, mean - std, mean + std,
                           color=METHOD_COLORS[method_id], alpha=0.15)
        
        ax.set_title(task)
        ax.set_xlabel('Training steps (k)')
        ax.set_ylabel('Mean eval reward')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.axhline(y=SUCCESS_THRESHOLD, color='gray', linestyle='--', alpha=0.4, label='Success threshold')
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_learning_curves.png'))
    plt.close()
    print("  [1/8] Learning curves saved.")


# ─────────────────────────────────────────────────────────────
# ANALYSIS 2: Entropy coefficient trajectories
# THIS IS THE MOST IMPORTANT ANALYSIS
# ─────────────────────────────────────────────────────────────
def plot_entropy_trajectories(base_dir, out_dir):
    """
    For methods A and C (auto entropy): the ent_coef_log stores -1 sentinel.
    
    RECOVERY OPTIONS (try in order):
    1. If ent_coef_log.csv has actual values (not -1) -> use directly
    2. If you saved model checkpoints -> load model, read model.log_ent_coef
    3. If tensorboard logs exist -> parse the 'train/ent_coef' scalar
    4. If none of the above -> we can only compare B/D (known schedule)
       against the SUCCESS PATTERNS of A/C
    
    This function tries option 1, flags if recovery is needed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    has_auto_data = False
    
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        
        for method_id in ['A', 'B', 'C', 'D']:
            all_alpha = []
            all_steps = []
            
            for seed in SEEDS:
                run_dir = find_run_dir(base_dir, task, method_id, seed)
                if run_dir is None:
                    continue
                
                df = load_csv_safe(os.path.join(run_dir, 'ent_coef_log.csv'))
                if df is None:
                    continue
                
                # Check if this has real values or sentinel
                real_values = df[df['ent_coef'] > 0]
                if len(real_values) == 0:
                    continue
                
                if method_id in ['A', 'C'] and (real_values['ent_coef'] != -1.0).any():
                    has_auto_data = True
                
                # Filter out sentinels
                valid = df[df['ent_coef'] > 0].copy()
                all_alpha.append(valid['ent_coef'].values)
                all_steps.append(valid['step'].values)
            
            if not all_alpha:
                # No valid data for this method — add annotation
                if method_id in ['A', 'C']:
                    ax.text(0.5, 0.5 if method_id == 'A' else 0.3,
                           f"{method_id}: auto α (not logged — see recovery instructions)",
                           transform=ax.transAxes, fontsize=9, alpha=0.5,
                           ha='center', style='italic')
                continue
            
            # Align and aggregate
            min_len = min(len(a) for a in all_alpha)
            alphas = np.array([a[:min_len] for a in all_alpha])
            steps = all_steps[0][:min_len]
            
            mean_alpha = alphas.mean(axis=0)
            std_alpha = alphas.std(axis=0)
            
            ax.plot(steps / 1000, mean_alpha, color=METHOD_COLORS[method_id],
                    label=f"{method_id}: {METHOD_LABELS[method_id]}", linewidth=1.5)
            ax.fill_between(steps / 1000, mean_alpha - std_alpha, mean_alpha + std_alpha,
                           color=METHOD_COLORS[method_id], alpha=0.15)
        
        ax.set_title(f"{task} — entropy coefficient α")
        ax.set_xlabel('Training steps (k)')
        ax.set_ylabel('α (entropy coeff)')
        ax.set_yscale('log')
        ax.legend(loc='upper right', framealpha=0.8)
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_entropy_trajectories.png'))
    plt.close()
    
    if not has_auto_data:
        print("  [2/8] Entropy trajectories saved — WARNING: auto-α not logged!")
        print("         Run the RECOVERY SCRIPT below to extract from saved models.")
    else:
        print("  [2/8] Entropy trajectories saved (auto-α data found).")
    
    return has_auto_data


# ─────────────────────────────────────────────────────────────
# ANALYSIS 3: Per-seed solved vs unsolved comparison
# ─────────────────────────────────────────────────────────────
def analyze_solved_vs_unsolved(base_dir, out_dir):
    """
    Split seeds into SOLVED (final reward >= threshold) and UNSOLVED.
    Compare their trajectories across all logged metrics.
    """
    records = []
    
    for task in TASKS:
        for method_id in ['A', 'B', 'C', 'D']:
            for seed in SEEDS:
                run_dir = find_run_dir(base_dir, task, method_id, seed)
                if run_dir is None:
                    continue
                
                ts, results = load_eval_data(run_dir)
                if ts is None:
                    continue
                
                mean_rewards = results.mean(axis=1)
                
                # First success step
                fss_path = os.path.join(run_dir, 'first_success_step.txt')
                first_success = None
                if os.path.exists(fss_path):
                    try:
                        first_success = int(open(fss_path).read().strip())
                    except (ValueError, FileNotFoundError):
                        pass
                
                # Buffer success fraction at 500k
                buf_df = load_csv_safe(os.path.join(run_dir, 'buffer_success_log.csv'))
                buf_at_500k = None
                buf_at_1M = None
                if buf_df is not None:
                    row_500k = buf_df[buf_df['step'] == 500000]
                    row_1M = buf_df[buf_df['step'] == 1000000]
                    if len(row_500k) > 0:
                        buf_at_500k = row_500k['buffer_success_fraction'].values[0]
                    if len(row_1M) > 0:
                        buf_at_1M = row_1M['buffer_success_fraction'].values[0]
                
                # Final and peak reward
                final_reward = mean_rewards[-20:].mean() if len(mean_rewards) >= 20 else mean_rewards[-1]
                peak_reward = mean_rewards.max()
                solved = final_reward >= SUCCESS_THRESHOLD
                
                # Q-value trajectory summary
                qv_df = load_csv_safe(os.path.join(run_dir, 'qvalue_probe_log.csv'))
                q_final = None
                q_peak = None
                q_collapsed = False
                if qv_df is not None and len(qv_df) > 0:
                    valid_q = qv_df[qv_df['mean_q'] > 0]
                    if len(valid_q) > 0:
                        q_final = valid_q['mean_q'].iloc[-1]
                        q_peak = valid_q['mean_q'].max()
                        # Collapse = peak is >2x final (Q-values rose then fell)
                        if q_peak > 0 and q_final > 0:
                            q_collapsed = (q_peak / q_final) > 2.0
                
                records.append({
                    'task': task,
                    'method': method_id,
                    'method_name': METHOD_LABELS[method_id],
                    'seed': seed,
                    'final_reward': round(final_reward, 1),
                    'peak_reward': round(peak_reward, 1),
                    'solved': solved,
                    'first_success_step': first_success,
                    'buf_success_500k': buf_at_500k,
                    'buf_success_1M': buf_at_1M,
                    'q_final': q_final,
                    'q_peak': q_peak,
                    'q_collapsed': q_collapsed,
                })
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'seed_summary.csv'), index=False)
    
    # Print summary table
    print("\n  === SEED-LEVEL SUMMARY ===")
    for task in TASKS:
        print(f"\n  {task}:")
        task_df = df[df['task'] == task]
        for method_id in ['A', 'B', 'C', 'D']:
            m_df = task_df[task_df['method'] == method_id]
            n_solved = m_df['solved'].sum()
            mean_final = m_df['final_reward'].mean()
            std_final = m_df['final_reward'].std()
            n_q_collapse = m_df['q_collapsed'].sum()
            mean_fss = m_df[m_df['first_success_step'].notna()]['first_success_step'].mean()
            print(f"    {method_id} ({METHOD_LABELS[method_id]:20s}): "
                  f"final={mean_final:7.0f}±{std_final:6.0f}  "
                  f"solved={n_solved}/8  "
                  f"q_collapse={n_q_collapse}/8  "
                  f"avg_first_success={mean_fss:7.0f}k" if not np.isnan(mean_fss)
                  else f"    {method_id} ({METHOD_LABELS[method_id]:20s}): "
                       f"final={mean_final:7.0f}±{std_final:6.0f}  "
                       f"solved={n_solved}/8  "
                       f"q_collapse={n_q_collapse}/8  "
                       f"avg_first_success=never")
    
    print(f"\n  [3/8] Seed summary saved to seed_summary.csv")
    return df


# ─────────────────────────────────────────────────────────────
# ANALYSIS 4: Buffer success fraction over time
# ─────────────────────────────────────────────────────────────
def plot_buffer_success(base_dir, out_dir):
    """Buffer success fraction over training — does the buffer stay enriched?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        for method_id in ['A', 'B', 'C', 'D']:
            all_fracs = []
            common_steps = None
            
            for seed in SEEDS:
                run_dir = find_run_dir(base_dir, task, method_id, seed)
                if run_dir is None:
                    continue
                df = load_csv_safe(os.path.join(run_dir, 'buffer_success_log.csv'))
                if df is None or len(df) == 0:
                    continue
                all_fracs.append(df['buffer_success_fraction'].values)
                if common_steps is None:
                    common_steps = df['step'].values
            
            if not all_fracs:
                continue
            
            min_len = min(len(f) for f in all_fracs)
            fracs = np.array([f[:min_len] for f in all_fracs])
            steps = common_steps[:min_len]
            
            mean_f = fracs.mean(axis=0)
            std_f = fracs.std(axis=0)
            
            ax.plot(steps / 1000, mean_f, color=METHOD_COLORS[method_id],
                    label=f"{method_id}: {METHOD_LABELS[method_id]}", linewidth=1.5)
            ax.fill_between(steps / 1000, mean_f - std_f, np.clip(mean_f + std_f, 0, 1),
                           color=METHOD_COLORS[method_id], alpha=0.15)
        
        ax.set_title(f"{task} — replay buffer success fraction")
        ax.set_xlabel('Training steps (k)')
        ax.set_ylabel('Fraction of buffer with reward ≥ 500')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_buffer_success.png'))
    plt.close()
    print("  [4/8] Buffer success fraction saved.")


# ─────────────────────────────────────────────────────────────
# ANALYSIS 5: Q-value probe trajectories
# ─────────────────────────────────────────────────────────────
def plot_qvalue_probes(base_dir, out_dir):
    """Q-values at probe states — do they stabilize or collapse?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        for method_id in ['A', 'B', 'C', 'D']:
            all_q = []
            common_steps = None
            
            for seed in SEEDS:
                run_dir = find_run_dir(base_dir, task, method_id, seed)
                if run_dir is None:
                    continue
                df = load_csv_safe(os.path.join(run_dir, 'qvalue_probe_log.csv'))
                if df is None or len(df) == 0:
                    continue
                # Filter valid Q-values (probes may not exist early)
                valid = df[df['n_probes'] > 0].copy()
                if len(valid) < 5:
                    continue
                all_q.append(valid['mean_q'].values)
                if common_steps is None or len(valid['step'].values) < len(common_steps):
                    common_steps = valid['step'].values
            
            if not all_q or common_steps is None:
                continue
            
            min_len = min(len(q) for q in all_q)
            min_len = min(min_len, len(common_steps))
            qs = np.array([q[:min_len] for q in all_q])
            steps = common_steps[:min_len]
            
            mean_q = qs.mean(axis=0)
            std_q = qs.std(axis=0)
            
            ax.plot(steps / 1000, mean_q, color=METHOD_COLORS[method_id],
                    label=f"{method_id}: {METHOD_LABELS[method_id]}", linewidth=1.5)
            ax.fill_between(steps / 1000, mean_q - std_q, mean_q + std_q,
                           color=METHOD_COLORS[method_id], alpha=0.15)
        
        ax.set_title(f"{task} — Q-value at probe states")
        ax.set_xlabel('Training steps (k)')
        ax.set_ylabel('Q(s_probe, a_probe)')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_qvalue_probes.png'))
    plt.close()
    print("  [5/8] Q-value probes saved.")


# ─────────────────────────────────────────────────────────────
# ANALYSIS 6: Policy entropy at near-object states
# ─────────────────────────────────────────────────────────────
def plot_policy_entropy(base_dir, out_dir):
    """Per-state entropy specifically at near-object states."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        any_data = False
        
        for method_id in ['A', 'B', 'C', 'D']:
            all_ent = []
            common_steps = None
            
            for seed in SEEDS:
                run_dir = find_run_dir(base_dir, task, method_id, seed)
                if run_dir is None:
                    continue
                df = load_csv_safe(os.path.join(run_dir, 'policy_entropy_log.csv'))
                if df is None or len(df) == 0:
                    continue
                # Filter out sentinel -1.0 values
                valid = df[df['mean_near_object_entropy'] > -0.5].copy()
                if len(valid) < 3:
                    continue
                all_ent.append(valid['mean_near_object_entropy'].values)
                if common_steps is None:
                    common_steps = valid['step'].values
                any_data = True
            
            if not all_ent:
                continue
            
            min_len = min(len(e) for e in all_ent)
            min_len = min(min_len, len(common_steps) if common_steps is not None else min_len)
            ents = np.array([e[:min_len] for e in all_ent])
            steps = common_steps[:min_len]
            
            mean_e = ents.mean(axis=0)
            ax.plot(steps / 1000, mean_e, color=METHOD_COLORS[method_id],
                    label=f"{method_id}", linewidth=1.5, marker='o', markersize=3)
        
        if not any_data:
            ax.text(0.5, 0.5, "Insufficient near-object entropy data\n(most seeds solved or never reached object)",
                   transform=ax.transAxes, ha='center', va='center', fontsize=10, alpha=0.5)
        
        ax.set_title(f"{task} — policy entropy at near-object states")
        ax.set_xlabel('Training steps (k)')
        ax.set_ylabel('H(π) at EE < 5cm from object')
        ax.legend(loc='upper right', framealpha=0.8)
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_policy_entropy.png'))
    plt.close()
    print("  [6/8] Policy entropy saved (may be sparse — see note in report).")


# ─────────────────────────────────────────────────────────────
# ANALYSIS 7: Peak vs Final scatter (retention diagnostic)
# ─────────────────────────────────────────────────────────────
def plot_peak_vs_final(base_dir, out_dir, df_summary):
    """
    Scatter: x = peak reward, y = final reward.
    Points on the diagonal = perfect retention.
    Points below = forgetting (peak > final).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        task_df = df_summary[df_summary['task'] == task]
        
        for method_id in ['A', 'B', 'C', 'D']:
            m_df = task_df[task_df['method'] == method_id]
            ax.scatter(m_df['peak_reward'], m_df['final_reward'],
                      c=METHOD_COLORS[method_id], label=f"{method_id}: {METHOD_LABELS[method_id]}",
                      s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Diagonal (perfect retention)
        max_val = max(task_df['peak_reward'].max(), task_df['final_reward'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
        ax.fill_between([0, max_val], [0, max_val], [0, 0],
                       color='red', alpha=0.03)
        ax.text(max_val * 0.7, max_val * 0.15, 'forgetting zone',
               fontsize=9, alpha=0.3, style='italic')
        
        # Success threshold lines
        ax.axhline(y=SUCCESS_THRESHOLD, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=SUCCESS_THRESHOLD, color='gray', linestyle=':', alpha=0.3)
        
        ax.set_title(f"{task} — peak vs final reward")
        ax.set_xlabel('Peak eval reward')
        ax.set_ylabel('Final eval reward (last-20 mean)')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '07_peak_vs_final.png'))
    plt.close()
    print("  [7/8] Peak vs Final scatter saved.")


# ─────────────────────────────────────────────────────────────
# ANALYSIS 8: First success step comparison
# ─────────────────────────────────────────────────────────────
def plot_first_success(base_dir, out_dir, df_summary):
    """Bar chart: time-to-first-success per method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        task_df = df_summary[df_summary['task'] == task]
        
        positions = []
        labels = []
        for mi, method_id in enumerate(['A', 'B', 'C', 'D']):
            m_df = task_df[task_df['method'] == method_id]
            fss = m_df['first_success_step'].dropna().values
            
            if len(fss) > 0:
                # Jittered strip plot
                jitter = np.random.uniform(-0.15, 0.15, len(fss))
                ax.scatter(np.full(len(fss), mi) + jitter, fss / 1000,
                          c=METHOD_COLORS[method_id], s=40, alpha=0.7,
                          edgecolors='white', linewidth=0.5)
                ax.plot([mi - 0.2, mi + 0.2], [fss.mean() / 1000] * 2,
                       color=METHOD_COLORS[method_id], linewidth=2)
            
            # Mark "never solved" seeds
            n_never = m_df['first_success_step'].isna().sum()
            if n_never > 0:
                ax.text(mi, ax.get_ylim()[1] if ax.get_ylim()[1] > 100 else 1050,
                       f'{n_never} never', ha='center', fontsize=8, alpha=0.5)
            
            positions.append(mi)
            labels.append(f"{method_id}")
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(f"{task} — time to first success")
        ax.set_ylabel('First success step (k)')
        ax.grid(alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '08_first_success.png'))
    plt.close()
    print("  [8/8] First success timing saved.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, 
                       default='experiments/exp35_causal_ablation/logs',
                       help='Directory containing the 64 run folders')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    out_dir = os.path.join(os.path.dirname(base_dir), 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Mining mechanistic logs from: {base_dir}")
    print(f"Output directory: {out_dir}")
    
    # Check directory exists and count runs
    if not os.path.isdir(base_dir):
        print(f"ERROR: {base_dir} does not exist!")
        sys.exit(1)
    
    run_count = 0
    for task in TASKS:
        for method_id in METHODS:
            for seed in SEEDS:
                if find_run_dir(base_dir, task, method_id, seed):
                    run_count += 1
    print(f"Found {run_count}/64 run directories.\n")
    
    # Run all analyses
    print("Running analyses...")
    plot_learning_curves(base_dir, out_dir)
    has_auto_alpha = plot_entropy_trajectories(base_dir, out_dir)
    df_summary = analyze_solved_vs_unsolved(base_dir, out_dir)
    plot_buffer_success(base_dir, out_dir)
    plot_qvalue_probes(base_dir, out_dir)
    plot_policy_entropy(base_dir, out_dir)
    plot_peak_vs_final(base_dir, out_dir, df_summary)
    plot_first_success(base_dir, out_dir, df_summary)
    
    # ─────────────────────────────────────────────────────────
    # RECOVERY INSTRUCTIONS for auto-α
    # ─────────────────────────────────────────────────────────
    if not has_auto_alpha:
        print("\n" + "="*70)
        print("CRITICAL: Auto-entropy α was NOT captured in logs!")
        print("="*70)
        print("""
You need to recover it. Here are your options, from best to worst:

OPTION 1 — Re-extract from saved model checkpoints (BEST)
If you saved model .zip files at checkpoints during training, run:

    import torch
    from stable_baselines3 import SAC
    
    model = SAC.load("path/to/checkpoint.zip")
    alpha = model.log_ent_coef.exp().item()
    print(f"α = {alpha}")

Loop over checkpoints to build the trajectory.

OPTION 2 — Re-run methods A and C with proper α logging (2-3 days)
Add this callback to your training script:

    class AlphaLogger(BaseCallback):
        def __init__(self, log_path):
            super().__init__()
            self.log_path = log_path
            self.records = []
        
        def _on_step(self):
            if self.num_timesteps % 1000 == 0:
                alpha = self.model.log_ent_coef.exp().item()
                self.records.append({
                    'step': self.num_timesteps,
                    'ent_coef': alpha
                })
            return True
        
        def _on_training_end(self):
            pd.DataFrame(self.records).to_csv(self.log_path, index=False)

Re-run only methods A and C (32 runs) with this callback.
Everything else stays identical.

OPTION 3 — Use indirect evidence (NO EXTRA COMPUTE)
Even without the α trajectory, you can still tell the story:
- Compare BUFFER SUCCESS FRACTION between A and B
- Compare Q-VALUE STABILITY between A and B  
- Compare LEARNING CURVE SHAPE (A rises smoothly, B/D plateau early)
These are indirect but sufficient for the paper.

RECOMMENDATION: Try Option 1 first (5 minutes if checkpoints exist).
If no checkpoints, go with Option 3 for now and submit to RA-L.
Option 2 for a revision if reviewers ask.
""")
    
    # Final diagnostic summary
    print("\n" + "="*70)
    print("DIAGNOSTIC QUESTIONS TO ANSWER FROM THESE PLOTS")
    print("="*70)
    print("""
Look at the plots and answer these questions:

Q1: Does method A (baseline) show RETENTION on peg-insert?
    → Look at 07_peak_vs_final.png: are A's points near the diagonal?
    → If YES: SAC already retains well. Your thesis needs revision.

Q2: Does method B show EARLY PLATEAU on pick-place?
    → Look at 01_learning_curves.png: does B flatline early?
    → If YES: the annealing schedule killed exploration too soon.

Q3: Does the buffer success fraction DIVERGE between A and B/D?
    → Look at 04_buffer_success.png: does A accumulate more successes?
    → If YES: annealing is preventing discovery, not improving retention.

Q4: Do Q-values collapse for any method?
    → Look at 05_qvalue_probes.png: spike then crash = forgetting.
    → If A's Q-values are stable: SAC auto-tunes well on its own.

Q5: Is there a SINGLE seed in B or D that outperforms all A seeds?
    → Look at seed_summary.csv: sort by final_reward descending.
    → If NO: annealing never helps, even in the best case.

Your answers determine the paper direction:
- If A retains AND B hurts → paper about "adaptive entropy > fixed schedule"
- If A retains AND C sometimes helps → paper about task-specific demo reward
- If nobody retains well → paper about the fundamental difficulty of retention
""")
    
    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == '__main__':
    main()