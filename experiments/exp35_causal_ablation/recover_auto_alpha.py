#!/usr/bin/env python3
"""
recover_auto_alpha.py
=====================
Extract the learned entropy coefficient α from SB3 SAC model checkpoints.

This is the MOST IMPORTANT analysis for your paper direction.
If SAC's auto-α naturally anneals on successful seeds, it explains
why your fixed annealing schedule hurts: it's a worse version of
what SAC already does adaptively.

Usage:
    python recover_auto_alpha.py --base_dir experiments/exp35_causal_ablation/logs

If you DON'T have model checkpoints:
    python recover_auto_alpha.py --base_dir ... --no-checkpoints
    (This extracts what it can from existing logs only)
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd

def try_load_alpha_from_checkpoint(checkpoint_path):
    """Try to extract α from a saved SB3 model."""
    try:
        import torch
        from stable_baselines3 import SAC
        
        # SB3 saves log_ent_coef; α = exp(log_ent_coef)
        model = SAC.load(checkpoint_path, device='cpu')
        
        if hasattr(model, 'log_ent_coef'):
            alpha = model.log_ent_coef.exp().item()
            return alpha
        else:
            return None
    except Exception as e:
        return None


def scan_for_checkpoints(run_dir):
    """Find all model checkpoint files in a run directory."""
    patterns = [
        os.path.join(run_dir, '*.zip'),
        os.path.join(run_dir, 'models', '*.zip'),
        os.path.join(run_dir, 'checkpoints', '*.zip'),
        os.path.join(run_dir, 'model_*.zip'),
        os.path.join(run_dir, 'best_model.zip'),
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    return sorted(found)


def extract_step_from_filename(filename):
    """Try to extract training step from checkpoint filename."""
    import re
    basename = os.path.basename(filename)
    # Common patterns: model_500000.zip, checkpoint_500k.zip, sac_500000_steps.zip
    matches = re.findall(r'(\d+)', basename)
    if matches:
        step = int(matches[-1])  # Take the last number
        # If it's small, might be in k
        if step < 10000:
            step *= 1000
        return step
    return None


def analyze_ent_coef_logs_indirect(base_dir):
    """
    When we can't recover auto-α directly, analyze what we CAN see:
    - For methods B/D: the annealed schedule IS logged
    - For methods A/C: compare their PERFORMANCE TRAJECTORY
      to B/D's known α trajectory
    
    Key insight: if A outperforms B despite B having the "better" 
    (lower) α after 500k, it means SAC's adaptive α was already
    finding the right level — and B's forced low α was too aggressive.
    """
    print("\n" + "="*60)
    print("INDIRECT α ANALYSIS (no checkpoints needed)")
    print("="*60)
    
    TASKS = ['peg-insert-side-v3', 'pick-place-v3']
    SEEDS = range(8)
    
    for task in TASKS:
        print(f"\n--- {task} ---")
        
        for method_id in ['A', 'B']:
            # Load all eval curves for this method
            solved_curves = []
            unsolved_curves = []
            
            for seed in SEEDS:
                # Find run dir
                patterns = [
                    f"{task}__method{method_id}__seed{seed}",
                    f"{task.replace('-', '_')}__method{method_id}__seed{seed}",
                ]
                run_dir = None
                for p in patterns:
                    path = os.path.join(base_dir, p)
                    if os.path.isdir(path):
                        run_dir = path
                        break
                
                if not run_dir:
                    matches = glob.glob(os.path.join(base_dir, 
                        f"*{task}*method*{method_id}*seed*{seed}*"))
                    if matches:
                        run_dir = matches[0]
                
                if not run_dir:
                    continue
                
                # Load eval
                npz_path = os.path.join(run_dir, 'eval', 'evaluations.npz')
                if not os.path.exists(npz_path):
                    continue
                data = np.load(npz_path)
                ts = data['timesteps']
                rewards = data['results'].mean(axis=1)
                
                # Check if solved
                final = rewards[-20:].mean() if len(rewards) >= 20 else rewards[-1]
                if final >= 500:
                    solved_curves.append((seed, rewards))
                else:
                    unsolved_curves.append((seed, rewards))
            
            n_solved = len(solved_curves)
            n_total = n_solved + len(unsolved_curves)
            
            print(f"\n  Method {method_id}: {n_solved}/{n_total} seeds solved")
            
            if n_solved > 0:
                # When do solved seeds "take off"?
                for seed, rewards in solved_curves:
                    # Find first checkpoint above threshold
                    above = np.where(rewards >= 500)[0]
                    if len(above) > 0:
                        first_idx = above[0]
                        step = ts[first_idx] if first_idx < len(ts) else 'unknown'
                        final_r = rewards[-5:].mean()
                        print(f"    Seed {seed}: first success at step {step/1000:.0f}k, "
                              f"final reward: {final_r:.0f}")
                    else:
                        print(f"    Seed {seed}: gradual learning (never crossed {500} cleanly)")
            
            if len(unsolved_curves) > 0:
                for seed, rewards in unsolved_curves:
                    peak = rewards.max()
                    final_r = rewards[-5:].mean()
                    print(f"    Seed {seed} (UNSOLVED): peak={peak:.0f}, final={final_r:.0f}")
        
        # The KEY comparison
        print(f"\n  KEY QUESTION: Does B's forced α=0.005 (after 500k) hurt vs A's auto-α?")
        print(f"  → If A has MORE solved seeds with HIGHER final reward:")
        print(f"     SAC's adaptive entropy was already doing the right thing.")
        print(f"  → If B solved seeds have BETTER retention (closer to peak):")
        print(f"     Annealing helps retention but hurts discovery.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                       default='experiments/exp35_causal_ablation/logs')
    parser.add_argument('--no-checkpoints', action='store_true',
                       help='Skip checkpoint loading, do indirect analysis only')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    if not args.no_checkpoints:
        # Try to find and load checkpoints
        print("Scanning for model checkpoints...")
        TASKS = ['peg-insert-side-v3', 'pick-place-v3']
        
        total_found = 0
        alpha_records = []
        
        for task in TASKS:
            for method_id in ['A', 'C']:  # Only auto-entropy methods
                for seed in range(8):
                    # Find run dir
                    patterns = [
                        f"{task}__method{method_id}__seed{seed}",
                        f"{task.replace('-', '_')}__method{method_id}__seed{seed}",
                    ]
                    run_dir = None
                    for p in patterns:
                        path = os.path.join(base_dir, p)
                        if os.path.isdir(path):
                            run_dir = path
                            break
                    
                    if not run_dir:
                        matches = glob.glob(os.path.join(base_dir,
                            f"*{task}*method*{method_id}*seed*{seed}*"))
                        if matches:
                            run_dir = matches[0]
                    
                    if not run_dir:
                        continue
                    
                    checkpoints = scan_for_checkpoints(run_dir)
                    if checkpoints:
                        total_found += len(checkpoints)
                        print(f"  {task}/method{method_id}/seed{seed}: "
                              f"found {len(checkpoints)} checkpoints")
                        
                        for cp in checkpoints:
                            step = extract_step_from_filename(cp)
                            alpha = try_load_alpha_from_checkpoint(cp)
                            if alpha is not None:
                                alpha_records.append({
                                    'task': task,
                                    'method': method_id,
                                    'seed': seed,
                                    'step': step,
                                    'alpha': alpha,
                                    'checkpoint': os.path.basename(cp)
                                })
        
        if total_found == 0:
            print("\nNo model checkpoints found.")
            print("This is expected if you didn't save models during training.")
            print("Falling back to indirect analysis...\n")
        elif alpha_records:
            df = pd.DataFrame(alpha_records)
            out_path = os.path.join(os.path.dirname(base_dir), 
                                    'analysis', 'recovered_alpha.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"\nRecovered α from {len(alpha_records)} checkpoints!")
            print(f"Saved to: {out_path}")
            
            # Quick summary
            print("\nAuto-α summary:")
            for task in TASKS:
                for method_id in ['A', 'C']:
                    subset = df[(df['task'] == task) & (df['method'] == method_id)]
                    if len(subset) > 0:
                        print(f"  {task} / Method {method_id}:")
                        for _, row in subset.iterrows():
                            print(f"    Step {row['step']}: α = {row['alpha']:.6f}")
            return
    
    # If no checkpoints or --no-checkpoints flag
    analyze_ent_coef_logs_indirect(base_dir)
    
    print("\n" + "="*60)
    print("WHAT TO DO NEXT")
    print("="*60)
    print("""
Based on your exp35 results (A > B > D > C on peg-insert):

1. RUN THIS ANALYSIS SCRIPT to understand the mechanism
   
2. If the analysis confirms SAC auto-tunes well:
   
   YOUR NEW PAPER THESIS becomes one of:
   
   (a) "Fixed entropy schedules hurt manipulation RL: 
        SAC's adaptive α already provides task-appropriate annealing"
        → Publishable negative result with mechanistic explanation
        → Target: RA-L (rolling), IROS workshop, CoRL workshop
   
   (b) "When does reward shaping help robotic manipulation RL?
        A controlled study across task reward density"
        → Run additional experiments on SPARSE reward tasks
        → If demo reward helps on sparse but hurts on dense, 
          that's a strong contribution
        → Target: CoRL 2026, RA-L
   
   (c) Combine (a) and (b) into a single paper:
        "Adaptive entropy and reward density: understanding when
         modifications to SAC help robotic manipulation"
        → The 64-run ablation + sparse-reward extension
        → Target: CoRL 2026 (ambitious), RA-L (safe)

3. IMMEDIATE ACTION: 
   Run the main analysis script first:
     python exp35_mechanistic_analysis.py --base_dir {base_dir}
   
   Then answer the 5 diagnostic questions it prints.
   Send me the answers and I'll tell you exactly which paper to write.
""")


if __name__ == '__main__':
    main()