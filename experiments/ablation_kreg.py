"""
Ablation Study: k_reg Sensitivity for EntropyRAPS

Tests k_reg ∈ {1, 2, 3, 4, 5} on PathMNIST (balanced), 1 seed.

Addresses reviewer question: "How sensitive is ES-RAPS to k_reg?"

Usage:
    python experiments/ablation_kreg.py
    python experiments/ablation_kreg.py --dataset organamnist
    python experiments/ablation_kreg.py --seed 42
"""

import torch
import yaml
import numpy as np
import random
import sys
import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import entropy

sys.path.append('.')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model
from models.conformal_wrapper import (
    EntropyStratifiedRAPS, SizeOptimizedRAPS, predict_with_sets
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_checkpoint(model, dataset_name, device):
    """Try multiple checkpoint paths."""
    checkpoint_paths = [
        Path(f'/content/drive/MyDrive/MedConformal_Final_Submission5/{dataset_name}_final/models/best_model.pth'),
        Path(f'/content/drive/MyDrive/MedConformal_Final_Submission2/{dataset_name}_final/models/best_model.pth'),
        Path(f'./outputs/{dataset_name}_final/models/best_model.pth'),
    ]
    
    for ckpt in checkpoint_paths:
        if ckpt.exists():
            print(f"   Loading checkpoint: {ckpt}")
            checkpoint = torch.load(ckpt, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            return True
    
    print("   ⚠️ No checkpoint found!")
    return False


def compute_hard_coverage(probs, sets, labels, percentile=0.90):
    """Compute coverage on hard cases (top entropy percentile)."""
    ent = entropy(probs, axis=1)
    hard_threshold = np.quantile(ent, percentile)
    hard_mask = ent > hard_threshold
    
    if hard_mask.sum() == 0:
        return float('nan')
    
    hard_cov = np.mean([
        labels[i] in sets[i] 
        for i in range(len(labels)) if hard_mask[i]
    ])
    return hard_cov


def main():
    parser = argparse.ArgumentParser(description='k_reg ablation for EntropyRAPS')
    parser.add_argument('--dataset', type=str, default='pathmnist')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    
    # Output directory
    if args.save_dir is None:
        drive_path = Path('/content/drive/MyDrive/MedConformal_Final_Submission5')
        if drive_path.exists():
            save_dir = drive_path / 'ablation_results'
        else:
            save_dir = Path('./ablation_results')
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"k_reg ABLATION STUDY — EntropyRAPS")
    print(f"Dataset: {args.dataset}, Seed: {args.seed}, α: {args.alpha}")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['data']['dataset'] = args.dataset
    
    data_loader = MedMNISTDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    # Load model
    print("\n[2/3] Loading model...")
    model = get_model(config, device)
    if not load_model_checkpoint(model, args.dataset, device):
        print("Cannot proceed without model checkpoint.")
        return
    
    # 3-way split
    tune_ds, calib_ds, val_split_ds = data_loader.split_validation_3way(
        val_loader.dataset, pct_tune=0.3, pct_calib=0.4, pct_val=0.3
    )
    tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32, shuffle=False)
    val_split_loader = torch.utils.data.DataLoader(val_split_ds, batch_size=32, shuffle=False)
    
    # Run ablation
    print("\n[3/3] Running k_reg ablation...")
    k_reg_values = [1, 2, 3, 4, 5]
    results = []
    
    for k_reg in k_reg_values:
        print(f"\n{'─'*60}")
        print(f"  k_reg = {k_reg}")
        print(f"{'─'*60}")
        
        set_seed(args.seed)  # Reset seed for each k_reg
        
        # EntropyRAPS with this k_reg
        entropy_raps = EntropyStratifiedRAPS(
            model, tune_loader, calib_loader, val_split_loader,
            alpha=args.alpha, k_reg=k_reg, randomized=True, device=device
        )
        
        # Also run RAPS-Std for comparison
        raps_std = SizeOptimizedRAPS(
            model, tune_loader, calib_loader, val_split_loader,
            alpha=args.alpha, k_reg=k_reg, randomized=True, device=device
        )
        
        # Evaluate EntropyRAPS
        _, _, probs_e, sets_e, labels_e = predict_with_sets(entropy_raps, test_loader)
        cov_e = np.mean([labels_e[i] in sets_e[i] for i in range(len(labels_e))])
        size_e = np.mean([len(s) for s in sets_e])
        hard_cov_e = compute_hard_coverage(probs_e, sets_e, labels_e)
        
        # Evaluate RAPS-Std
        _, _, probs_r, sets_r, labels_r = predict_with_sets(raps_std, test_loader)
        cov_r = np.mean([labels_r[i] in sets_r[i] for i in range(len(labels_r))])
        size_r = np.mean([len(s) for s in sets_r])
        hard_cov_r = compute_hard_coverage(probs_r, sets_r, labels_r)
        
        results.append({
            'k_reg': k_reg,
            'Method': 'EntropyRAPS',
            'lambda_star': entropy_raps.lamda_star,
            'q_hat': entropy_raps.q_hat_star,
            'Coverage': cov_e,
            'Hard_Coverage': hard_cov_e,
            'Avg_Size': size_e,
        })
        results.append({
            'k_reg': k_reg,
            'Method': 'RAPS_Standard',
            'lambda_star': raps_std.lamda_star,
            'q_hat': raps_std.q_hat_star,
            'Coverage': cov_r,
            'Hard_Coverage': hard_cov_r,
            'Avg_Size': size_r,
        })
        
        print(f"  EntropyRAPS: λ*={entropy_raps.lamda_star:.5f}, "
              f"Cov={cov_e:.4f}, Hard={hard_cov_e:.4f}, Size={size_e:.2f}")
        print(f"  RAPS-Std:    λ*={raps_std.lamda_star:.5f}, "
              f"Cov={cov_r:.4f}, Hard={hard_cov_r:.4f}, Size={size_r:.2f}")
    
    # Format results
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("k_reg ABLATION RESULTS")
    print("=" * 70)
    
    # Pivot for clean display
    for method in ['EntropyRAPS', 'RAPS_Standard']:
        sub = df[df['Method'] == method]
        print(f"\n{method}:")
        print(f"{'k_reg':<8} {'λ*':<10} {'Coverage':<12} {'Hard Cov':<12} {'Avg Size':<10}")
        print("-" * 55)
        for _, row in sub.iterrows():
            print(f"{row['k_reg']:<8} {row['lambda_star']:<10.5f} "
                  f"{row['Coverage']:<12.4f} {row['Hard_Coverage']:<12.4f} "
                  f"{row['Avg_Size']:<10.2f}")
    
    # LaTeX table
    print("\n\n% LaTeX table (EntropyRAPS only):")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{k_reg sensitivity on {args.dataset} ($\\alpha={args.alpha}$)}}")
    print("\\begin{tabular}{ccccc}")
    print("\\toprule")
    print("$k_{\\text{reg}}$ & $\\lambda^*$ & Coverage & Hard Cov. & Avg. Size \\\\")
    print("\\midrule")
    sub_e = df[df['Method'] == 'EntropyRAPS']
    for _, row in sub_e.iterrows():
        print(f"{int(row['k_reg'])} & {row['lambda_star']:.4f} & "
              f"{row['Coverage']:.4f} & {row['Hard_Coverage']:.4f} & "
              f"{row['Avg_Size']:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save
    csv_path = save_dir / f'ablation_kreg_{args.dataset}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()