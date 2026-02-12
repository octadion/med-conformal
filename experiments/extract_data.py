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
from torch.utils.data import Subset, DataLoader as DL

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

class ImbalancedDifficultyDataset:
    def __init__(self, base_dataset, model, 
                 hard_ratio=0.1, medium_ratio=0.2, easy_ratio=0.9,
                 device='cuda', seed=42):
        
        self.base_dataset = base_dataset
        self.model = model
        self.device = device
        
        print(f"[Imbalanced Dataset] Creating difficulty imbalance...")
        print(f"   Target ratios - Easy: {easy_ratio:.0%}, Medium: {medium_ratio:.0%}, Hard: {hard_ratio:.0%}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        entropies = self._compute_entropy_for_dataset()
        
        t1, t2 = np.quantile(entropies, [0.33, 0.66])
        
        easy_mask = entropies <= t1
        medium_mask = (entropies > t1) & (entropies <= t2)
        hard_mask = entropies > t2
        
        easy_indices = np.where(easy_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        hard_indices = np.where(hard_mask)[0]
        
        print(f"   Original counts - Easy: {len(easy_indices)}, Medium: {len(medium_indices)}, Hard: {len(hard_indices)}")
        
        n_easy = int(len(easy_indices) * easy_ratio)
        n_medium = int(len(medium_indices) * medium_ratio)
        n_hard = int(len(hard_indices) * hard_ratio)
        
        selected_easy = np.random.choice(easy_indices, n_easy, replace=False)
        selected_medium = np.random.choice(medium_indices, n_medium, replace=False)
        selected_hard = np.random.choice(hard_indices, n_hard, replace=False)
        
        self.selected_indices = np.concatenate([selected_easy, selected_medium, selected_hard])
        np.random.shuffle(self.selected_indices)
        
        self.easy_mask_new = np.isin(self.selected_indices, selected_easy)
        self.medium_mask_new = np.isin(self.selected_indices, selected_medium)
        self.hard_mask_new = np.isin(self.selected_indices, selected_hard)
        
        print(f"   Imbalanced counts - Easy: {n_easy}, Medium: {n_medium}, Hard: {n_hard}")
        print(f"   New distribution - Easy: {n_easy/(n_easy+n_medium+n_hard):.1%}, "
              f"Medium: {n_medium/(n_easy+n_medium+n_hard):.1%}, "
              f"Hard: {n_hard/(n_easy+n_medium+n_hard):.1%}")
    
    def _compute_entropy_for_dataset(self):
        self.model.eval()
        entropies = []
        
        loader = torch.utils.data.DataLoader(self.base_dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_entropy = entropy(probs, axis=1)
                entropies.extend(batch_entropy)
        
        return np.array(entropies)
    
    def get_subset(self):
        return Subset(self.base_dataset, self.selected_indices)
    
    def get_stratum_info(self):
        return {
            'easy_count': self.easy_mask_new.sum(),
            'medium_count': self.medium_mask_new.sum(),
            'hard_count': self.hard_mask_new.sum(),
        }


def extract_imbalanced_strata_sizes(device='cuda', save_dir='.'):
    print("\n" + "=" * 70)
    print("DIVERGENCE EXPERIMENT (Extended with Mondrian methods)")
    print("=" * 70)
    
    dataset_name = 'pathmnist'
    model, data_loader, val_loader, test_loader, config = \
        setup_data_and_model(dataset_name, device)
    if model is None:
        return
    
    # === EXACT SAME AS ORIGINAL ===
    
    # Create IMBALANCED datasets
    print("\n[1/4] Creating imbalanced difficulty datasets...")
    imb_val = ImbalancedDifficultyDataset(
        val_loader.dataset, model,
        hard_ratio=0.05, medium_ratio=0.1, easy_ratio=0.95,
        device=device, seed=42
    )
    imb_test = ImbalancedDifficultyDataset(
        test_loader.dataset, model,
        hard_ratio=0.05, medium_ratio=0.1, easy_ratio=0.95,
        device=device, seed=43
    )
    
    # 3-way split from imbalanced val
    print("\n[2/4] Creating 3-way split...")
    imb_val_ds = imb_val.get_subset()
    n = len(imb_val_ds)
    
    n_tune = int(n * 0.3)
    n_calib = int(n * 0.4)
    
    indices = np.random.permutation(n)
    tune_indices = indices[:n_tune]
    calib_indices = indices[n_tune:n_tune + n_calib]
    val_indices = indices[n_tune + n_calib:]
    
    tune_ds = Subset(imb_val_ds, tune_indices)
    calib_ds = Subset(imb_val_ds, calib_indices)
    val_ds = Subset(imb_val_ds, val_indices)
    
    tune_loader = DL(tune_ds, batch_size=32, shuffle=False)
    calib_loader = DL(calib_ds, batch_size=32, shuffle=False)
    val_split_loader = DL(val_ds, batch_size=32, shuffle=False)
    test_imb_loader = DL(imb_test.get_subset(), batch_size=32, shuffle=False)
    
    print(f"   Tune: {len(tune_ds)}, Calib: {len(calib_ds)}, Val: {len(val_ds)}")
    
    # Fit methods — original 2 + new 2
    print("\n[3/4] Running conformal methods...")
    
    methods = {}
    
    methods['RAPS-Std'] = SizeOptimizedRAPS(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=0.1, k_reg=2, randomized=True, device=device
    )
    methods['EntropyRAPS'] = EntropyStratifiedRAPS(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=0.1, k_reg=2, randomized=True, device=device
    )
    # NEW methods
    methods['ClassMondrian'] = ClassMondrianCP(
        model, calib_loader=calib_loader, alpha=0.1,
        randomized=True, min_class_size=5, device=device
    )
    methods['StratifiedCP'] = StratifiedCP(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=0.1, n_strata=3, device=device
    )
    
    # === DIVERGENCE CHECK (same as original) ===
    print(f"\n{'='*60}")
    print("DIVERGENCE RESULTS:")
    print(f"{'='*60}")
    print(f"Standard RAPS λ: {methods['RAPS-Std'].lamda_star:.5f}")
    print(f"EntropyRAPS λ:   {methods['EntropyRAPS'].lamda_star:.5f}")
    print(f"Difference:      {abs(methods['RAPS-Std'].lamda_star - methods['EntropyRAPS'].lamda_star):.5f}")
    
    if abs(methods['RAPS-Std'].lamda_star - methods['EntropyRAPS'].lamda_star) > 0.01:
        print("\n✅ DIVERGENCE DETECTED!")
    else:
        print("\n⚠️ No significant divergence")
    
    # === EVALUATION (same hard-case definition as original) ===
    print(f"\n[4/4] Evaluating...")
    print(f"\n{'='*60}")
    print("COVERAGE COMPARISON:")
    print(f"{'='*60}")
    
    results = []
    
    # Evaluate all methods, compute hard-case same way as original
    
    print(f"\n{'Method':<18} {'Overall Cov':<15} {'Hard Cov':<15} {'Avg Size':<10}")
    print("-" * 60)
    
    for method_name, conf_model in methods.items():
        _, _, probs, sets, labels = predict_with_sets(conf_model, test_imb_loader)
        
        cov = np.mean([labels[i] in sets[i] for i in range(len(labels))])
        avg_size = np.mean([len(s) for s in sets])
        
        # Hard-case: top 10% entropy from test probs (SAME as original)
        ent_vals = entropy(probs, axis=1)
        hard_mask = ent_vals > np.quantile(ent_vals, 0.9)
        hard_cov = np.mean([labels[i] in sets[i] for i in range(len(labels)) if hard_mask[i]])
        
        print(f"{method_name:<18} {cov:<15.1%} {hard_cov:<15.1%} {avg_size:<10.2f}")
        
        results.append({
            'Method': method_name,
            'Overall_Coverage': cov,
            'Hard_Coverage': hard_cov,
            'Avg_Size': avg_size,
            'lambda_star': getattr(conf_model, 'lamda_star', None),
        })
    
    print("-" * 60)
    
    # Hard-case improvement (EntropyRAPS vs RAPS-Std, same as original)
    raps_hard = results[0]['Hard_Coverage']
    entropy_hard = results[1]['Hard_Coverage']
    print(f"\nHard-case coverage improvement (EntropyRAPS vs RAPS): {entropy_hard - raps_hard:+.1%}")
    if entropy_hard > raps_hard:
        print("✅ EntropyRAPS improves hard-case coverage!")
    
    # LaTeX
    print(f"\n% LaTeX:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Divergence experiment on imbalanced PathMNIST "
          "(95\\% easy, 10\\% med, 5\\% hard)}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Method & Overall Cov. & Hard Cov. & Avg. Size \\\\")
    print("\\midrule")
    for r in results:
        lam = r['lambda_star']
        lam_str = f" ($\\lambda^*$={lam:.4f})" if lam is not None else ""
        print(f"{r['Method']}{lam_str} & {r['Overall_Coverage']:.4f} & "
              f"{r['Hard_Coverage']:.4f} & {r['Avg_Size']:.2f} \\\\")
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