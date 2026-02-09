"""
Data Extraction: λ* Stability, Entropy Boundaries, Per-Stratum Sizes

Extracts paper-ready data from multi-seed runs:
  1. λ* stability across 5 seeds (mean ± std)
  2. Entropy boundary stability across 5 seeds
  3. Per-stratum set sizes for imbalanced experiment

Usage:
    python experiments/extract_data.py --all
    python experiments/extract_data.py --lambda_stability
    python experiments/extract_data.py --entropy_boundaries
    python experiments/extract_data.py --imbalanced_strata
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
    EntropyStratifiedRAPS, SizeOptimizedRAPS, ClassMondrianCP,
    StratifiedCP, StandardLAC, predict_with_sets,
    get_logits_labels, calculate_entropy
)


SEEDS = [42, 10, 2024, 99, 123]
DATASETS = ['pathmnist', 'organamnist']


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_checkpoint(model, dataset_name, device):
    """Try multiple checkpoint paths."""
    paths = [
        Path(f'/content/drive/MyDrive/MedConformal_Final_Submission5/{dataset_name}_final/models/best_model.pth'),
        Path(f'/content/drive/MyDrive/MedConformal_Final_Submission2/{dataset_name}_final/models/best_model.pth'),
        Path(f'./outputs/{dataset_name}_final/models/best_model.pth'),
    ]
    for p in paths:
        if p.exists():
            checkpoint = torch.load(p, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"   Loaded: {p}")
            return True
    print("   ⚠️ No checkpoint found!")
    return False


def setup_data_and_model(dataset_name, device):
    """Load model and data, return (model, data_loader_obj, config)."""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['data']['dataset'] = dataset_name
    
    data_loader = MedMNISTDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    model = get_model(config, device)
    if not load_model_checkpoint(model, dataset_name, device):
        return None, None, None, None, None
    
    return model, data_loader, val_loader, test_loader, config


# =============================================================================
# 1. λ* STABILITY ACROSS SEEDS
# =============================================================================

def extract_lambda_stability(device='cuda', save_dir='.'):
    """
    Run EntropyRAPS and RAPS-Std across 5 seeds, extract λ* values.
    
    Output:
        Dataset      | EntropyRAPS λ* (mean ± std) | RAPS-Std λ* (mean ± std)
        -------------|-----------------------------|--------------------------
        PathMNIST    | X.XXX ± X.XXX               | X.XXX ± X.XXX
        OrganAMNIST  | X.XXX ± X.XXX               | X.XXX ± X.XXX
    """
    print("\n" + "=" * 70)
    print("EXTRACTING: λ* Stability Across Seeds")
    print("=" * 70)
    
    all_results = []
    
    for dataset_name in DATASETS:
        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*60}")
        
        model, data_loader, val_loader, test_loader, config = \
            setup_data_and_model(dataset_name, device)
        if model is None:
            continue
        
        lambda_entropy = []
        lambda_raps = []
        
        for seed in SEEDS:
            set_seed(seed)
            
            tune_ds, calib_ds, val_split_ds = data_loader.split_validation_3way(
                val_loader.dataset, pct_tune=0.3, pct_calib=0.4, pct_val=0.3
            )
            tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
            calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32, shuffle=False)
            val_split_loader = torch.utils.data.DataLoader(val_split_ds, batch_size=32, shuffle=False)
            
            # EntropyRAPS
            e_raps = EntropyStratifiedRAPS(
                model, tune_loader, calib_loader, val_split_loader,
                alpha=0.1, k_reg=2, randomized=True, device=device
            )
            lambda_entropy.append(e_raps.lamda_star)
            
            # RAPS-Std
            r_std = SizeOptimizedRAPS(
                model, tune_loader, calib_loader, val_split_loader,
                alpha=0.1, k_reg=2, randomized=True, device=device
            )
            lambda_raps.append(r_std.lamda_star)
            
            print(f"   Seed {seed}: EntropyRAPS λ*={e_raps.lamda_star:.5f}, "
                  f"RAPS-Std λ*={r_std.lamda_star:.5f}")
        
        all_results.append({
            'Dataset': dataset_name,
            'EntropyRAPS_lambda_mean': np.mean(lambda_entropy),
            'EntropyRAPS_lambda_std': np.std(lambda_entropy),
            'RAPS_Std_lambda_mean': np.mean(lambda_raps),
            'RAPS_Std_lambda_std': np.std(lambda_raps),
            'EntropyRAPS_lambdas': lambda_entropy,
            'RAPS_Std_lambdas': lambda_raps,
        })
    
    # Print results
    print(f"\n{'='*70}")
    print("λ* STABILITY RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'EntropyRAPS λ* (mean±std)':<30} {'RAPS-Std λ* (mean±std)':<30}")
    print("-" * 75)
    for r in all_results:
        e_str = f"{r['EntropyRAPS_lambda_mean']:.5f} ± {r['EntropyRAPS_lambda_std']:.5f}"
        r_str = f"{r['RAPS_Std_lambda_mean']:.5f} ± {r['RAPS_Std_lambda_std']:.5f}"
        print(f"{r['Dataset']:<15} {e_str:<30} {r_str:<30}")
    
    # LaTeX
    print(f"\n% LaTeX:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{$\\lambda^*$ stability across 5 seeds}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Dataset & ES-RAPS $\\lambda^*$ & RAPS-Std $\\lambda^*$ \\\\")
    print("\\midrule")
    for r in all_results:
        print(f"{r['Dataset']} & "
              f"${r['EntropyRAPS_lambda_mean']:.4f} \\pm {r['EntropyRAPS_lambda_std']:.4f}$ & "
              f"${r['RAPS_Std_lambda_mean']:.4f} \\pm {r['RAPS_Std_lambda_std']:.4f}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save
    df = pd.DataFrame(all_results)
    csv_path = Path(save_dir) / 'lambda_stability.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved to: {csv_path}")
    
    return all_results


# =============================================================================
# 2. ENTROPY BOUNDARY STABILITY
# =============================================================================

def extract_entropy_boundaries(device='cuda', save_dir='.'):
    """
    Report entropy tertile thresholds across 5 seeds.
    
    Output:
        Dataset     | Low Threshold (mean ± std) | High Threshold (mean ± std)
        ------------|----------------------------|-----------------------------
        PathMNIST   | X.XXX ± X.XXX              | X.XXX ± X.XXX
        OrganAMNIST | X.XXX ± X.XXX              | X.XXX ± X.XXX
    """
    print("\n" + "=" * 70)
    print("EXTRACTING: Entropy Boundary Stability")
    print("=" * 70)
    
    all_results = []
    
    for dataset_name in DATASETS:
        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*60}")
        
        model, data_loader, val_loader, test_loader, config = \
            setup_data_and_model(dataset_name, device)
        if model is None:
            continue
        
        th1_values = []
        th2_values = []
        
        for seed in SEEDS:
            set_seed(seed)
            
            tune_ds, calib_ds, val_split_ds = data_loader.split_validation_3way(
                val_loader.dataset, pct_tune=0.3, pct_calib=0.4, pct_val=0.3
            )
            tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
            
            # Compute entropy boundaries on tune set
            logits_tune, _ = get_logits_labels(model, tune_loader, device)
            tune_entropy = calculate_entropy(logits_tune)
            th1, th2 = np.quantile(tune_entropy, [0.33, 0.66])
            
            th1_values.append(th1)
            th2_values.append(th2)
            
            print(f"   Seed {seed}: th1={th1:.4f}, th2={th2:.4f}")
        
        all_results.append({
            'Dataset': dataset_name,
            'th1_mean': np.mean(th1_values),
            'th1_std': np.std(th1_values),
            'th2_mean': np.mean(th2_values),
            'th2_std': np.std(th2_values),
        })
    
    # Print
    print(f"\n{'='*70}")
    print("ENTROPY BOUNDARY STABILITY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Low Threshold (mean±std)':<30} {'High Threshold (mean±std)':<30}")
    print("-" * 75)
    for r in all_results:
        lo = f"{r['th1_mean']:.4f} ± {r['th1_std']:.4f}"
        hi = f"{r['th2_mean']:.4f} ± {r['th2_std']:.4f}"
        print(f"{r['Dataset']:<15} {lo:<30} {hi:<30}")
    
    # LaTeX
    print(f"\n% LaTeX:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Entropy boundary stability across 5 seeds}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Dataset & Low threshold ($t_1$) & High threshold ($t_2$) \\\\")
    print("\\midrule")
    for r in all_results:
        print(f"{r['Dataset']} & "
              f"${r['th1_mean']:.4f} \\pm {r['th1_std']:.4f}$ & "
              f"${r['th2_mean']:.4f} \\pm {r['th2_std']:.4f}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    df = pd.DataFrame(all_results)
    csv_path = Path(save_dir) / 'entropy_boundaries.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved to: {csv_path}")
    
    return all_results


# =============================================================================
# 3. PER-STRATUM SET SIZES (IMBALANCED EXPERIMENT)
# =============================================================================

def extract_imbalanced_strata_sizes(device='cuda', save_dir='.'):
    """
    Extract per-stratum average set sizes for imbalanced PathMNIST experiment.
    
    Setup: 85% easy, 10% medium, 5% hard (same as divergence experiment)
    
    Output:
        Method       | Easy Size | Med Size | Hard Size | Overall
        -------------|-----------|----------|-----------|--------
        RAPS-Std     | ...       | ...      | ...       | ...
        EntropyRAPS  | ...       | ...      | ...       | ...
        ClassMondrian| ...       | ...      | ...       | ...
        StratifiedCP | ...       | ...      | ...       | ...
    """
    print("\n" + "=" * 70)
    print("EXTRACTING: Per-Stratum Set Sizes (Imbalanced PathMNIST)")
    print("=" * 70)
    
    dataset_name = 'pathmnist'
    model, data_loader, val_loader, test_loader, config = \
        setup_data_and_model(dataset_name, device)
    if model is None:
        return
    
    set_seed(42)
    
    # Create imbalanced test set
    from torch.utils.data import DataLoader as DL, Subset
    
    print("\n   Creating imbalanced difficulty distribution...")
    
    # Get test set entropy
    logits_test, labels_test = get_logits_labels(model, test_loader, device)
    test_entropy = calculate_entropy(logits_test)
    
    # Define strata boundaries (from full test set)
    t1, t2 = np.quantile(test_entropy, [0.33, 0.66])
    print(f"   Entropy boundaries: [{t1:.4f}, {t2:.4f}]")
    
    easy_mask = test_entropy <= t1
    medium_mask = (test_entropy > t1) & (test_entropy <= t2)
    hard_mask = test_entropy > t2
    
    easy_idx = np.where(easy_mask)[0]
    medium_idx = np.where(medium_mask)[0]
    hard_idx = np.where(hard_mask)[0]
    
    # Subsample to create imbalance: 85% easy, 10% medium, 5% hard
    n_easy = int(len(easy_idx) * 0.95)
    n_medium = int(len(medium_idx) * 0.10)
    n_hard = int(len(hard_idx) * 0.05)
    
    sel_easy = np.random.choice(easy_idx, n_easy, replace=False)
    sel_medium = np.random.choice(medium_idx, n_medium, replace=False)
    sel_hard = np.random.choice(hard_idx, n_hard, replace=False)
    
    total = n_easy + n_medium + n_hard
    print(f"   Imbalanced: Easy={n_easy} ({n_easy/total:.1%}), "
          f"Med={n_medium} ({n_medium/total:.1%}), "
          f"Hard={n_hard} ({n_hard/total:.1%})")
    
    imb_indices = np.concatenate([sel_easy, sel_medium, sel_hard])
    imb_ds = Subset(test_loader.dataset, imb_indices)
    imb_loader = DL(imb_ds, batch_size=32, shuffle=False)
    
    # Track which samples belong to which stratum
    imb_strata = np.zeros(len(imb_indices), dtype=int)
    imb_strata[:n_easy] = 0       # Easy
    imb_strata[n_easy:n_easy+n_medium] = 1  # Medium
    imb_strata[n_easy+n_medium:] = 2         # Hard
    
    # Setup conformal methods
    tune_ds, calib_ds, val_split_ds = data_loader.split_validation_3way(
        val_loader.dataset, pct_tune=0.3, pct_calib=0.4, pct_val=0.3
    )
    tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32, shuffle=False)
    val_split_loader = torch.utils.data.DataLoader(val_split_ds, batch_size=32, shuffle=False)
    
    methods = {}
    
    print("\n   Fitting methods...")
    methods['RAPS-Std'] = SizeOptimizedRAPS(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=0.1, k_reg=2, randomized=True, device=device
    )
    methods['EntropyRAPS'] = EntropyStratifiedRAPS(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=0.1, k_reg=2, randomized=True, device=device
    )
    methods['ClassMondrian'] = ClassMondrianCP(
        model, calib_loader=calib_loader, alpha=0.1,
        randomized=True, min_class_size=5, device=device
    )
    methods['StratifiedCP'] = StratifiedCP(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=0.1, n_strata=3, device=device
    )
    
    # Evaluate per-stratum
    results = []
    
    print(f"\n{'='*70}")
    print("PER-STRATUM SET SIZES (IMBALANCED)")
    print(f"{'='*70}")
    print(f"\n{'Method':<18} {'Easy Size':<12} {'Med Size':<12} {'Hard Size':<12} "
          f"{'Overall':<10} {'Easy Cov':<10} {'Med Cov':<10} {'Hard Cov':<10}")
    print("-" * 100)
    
    for method_name, conf_model in methods.items():
        _, _, probs, sets, labels = predict_with_sets(conf_model, imb_loader)
        
        sizes = np.array([len(s) for s in sets])
        is_covered = np.array([labels[i] in sets[i] for i in range(len(labels))])
        
        # Per-stratum metrics
        easy_sizes = sizes[imb_strata == 0]
        med_sizes = sizes[imb_strata == 1]
        hard_sizes = sizes[imb_strata == 2]
        
        easy_cov = np.mean(is_covered[imb_strata == 0])
        med_cov = np.mean(is_covered[imb_strata == 1])
        hard_cov = np.mean(is_covered[imb_strata == 2])
        
        overall_size = np.mean(sizes)
        overall_cov = np.mean(is_covered)
        
        print(f"{method_name:<18} {np.mean(easy_sizes):<12.2f} {np.mean(med_sizes):<12.2f} "
              f"{np.mean(hard_sizes):<12.2f} {overall_size:<10.2f} "
              f"{easy_cov:<10.4f} {med_cov:<10.4f} {hard_cov:<10.4f}")
        
        results.append({
            'Method': method_name,
            'Easy_Avg_Size': np.mean(easy_sizes),
            'Med_Avg_Size': np.mean(med_sizes),
            'Hard_Avg_Size': np.mean(hard_sizes),
            'Overall_Avg_Size': overall_size,
            'Easy_Coverage': easy_cov,
            'Med_Coverage': med_cov,
            'Hard_Coverage': hard_cov,
            'Overall_Coverage': overall_cov,
            'N_Easy': n_easy,
            'N_Medium': n_medium,
            'N_Hard': n_hard,
        })
    
    print("-" * 100)
    
    # LaTeX
    print(f"\n% LaTeX:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Per-stratum set sizes on imbalanced PathMNIST "
          "(85\\% easy, 10\\% med, 5\\% hard)}")
    print("\\begin{tabular}{lcccc|ccc}")
    print("\\toprule")
    print("Method & Easy & Med & Hard & Overall & Easy Cov & Med Cov & Hard Cov \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r['Method']} & {r['Easy_Avg_Size']:.2f} & {r['Med_Avg_Size']:.2f} & "
              f"{r['Hard_Avg_Size']:.2f} & {r['Overall_Avg_Size']:.2f} & "
              f"{r['Easy_Coverage']:.4f} & {r['Med_Coverage']:.4f} & "
              f"{r['Hard_Coverage']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    df = pd.DataFrame(results)
    csv_path = Path(save_dir) / 'imbalanced_strata_sizes.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved to: {csv_path}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract paper-ready data')
    parser.add_argument('--all', action='store_true', help='Run all extractions')
    parser.add_argument('--lambda_stability', action='store_true')
    parser.add_argument('--entropy_boundaries', action='store_true')
    parser.add_argument('--imbalanced_strata', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.save_dir is None:
        drive_path = Path('/content/drive/MyDrive/MedConformal_Final_Submission5')
        if drive_path.exists():
            save_dir = drive_path / 'extracted_data'
        else:
            save_dir = Path('./extracted_data')
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    run_all = args.all or not any([
        args.lambda_stability, args.entropy_boundaries, args.imbalanced_strata
    ])
    
    if run_all or args.lambda_stability:
        extract_lambda_stability(device, save_dir)
    
    if run_all or args.entropy_boundaries:
        extract_entropy_boundaries(device, save_dir)
    
    if run_all or args.imbalanced_strata:
        extract_imbalanced_strata_sizes(device, save_dir)
    
    print("\n" + "=" * 70)
    print(f"ALL EXTRACTIONS COMPLETE. Results in: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()