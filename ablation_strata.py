import torch
import torch.nn as nn
import yaml
import sys
from pathlib import Path
sys.path.append('.')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model
from models.conformal_wrapper import predict_with_sets
import numpy as np
import pandas as pd
from scipy.stats import entropy


def get_logits_labels(model, loader, device='cuda'):
    """Extract logits and labels from dataloader."""
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            logits_list.append(out.cpu())
            # FIX: Use view(-1) instead of squeeze()
            y = y.cpu().view(-1)
            labels_list.append(y)
    return torch.cat(logits_list), torch.cat(labels_list)


def calculate_entropy(logits):
    """Compute predictive entropy from logits."""
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return entropy(probs, axis=1)


def compute_raps_scores_randomized(logits, labels, lamda, k_reg, randomized=True):
    """
    Compute RAPS scores WITH randomization (matching original paper).
    
    FIXED: score = U * p_y + cumsum[rank-1] + penalty
    """
    probs = torch.softmax(logits, dim=1)
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    n_samples = len(labels)
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        rank_tensor = (indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        rank = rank_tensor.item() if len(rank_tensor) > 0 else 0
        
        p_y = sorted_probs[i, rank].item()
        penalty = lamda * max(rank - k_reg + 1, 0)
        
        if randomized:
            U = np.random.random()
        else:
            U = 1.0
        
        if rank == 0:
            scores[i] = U * p_y + penalty
        else:
            cumsum_before = cumsum[i, rank - 1].item()
            scores[i] = U * p_y + cumsum_before + penalty
    
    return scores


def compute_quantile(scores, alpha):
    """Compute conformal quantile."""
    return np.quantile(scores, 1 - alpha, method='higher')


def construct_prediction_sets(probs, q_hat, lamda, k_reg, num_classes, randomized=True):
    """Construct prediction sets using gcq algorithm."""
    n_samples = len(probs)
    
    I = probs.argsort(axis=1)[:, ::-1]
    ordered = np.sort(probs, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    
    penalties = np.zeros(num_classes)
    penalties[k_reg:] = lamda
    penalties_cumsum = np.cumsum(penalties)
    
    sizes_base = ((cumsum + penalties_cumsum) <= q_hat).sum(axis=1) + 1
    sizes_base = np.minimum(sizes_base, num_classes)
    
    if randomized:
        V = np.zeros(n_samples)
        for i in range(n_samples):
            idx = sizes_base[i] - 1
            if idx >= 0 and ordered[i, idx] > 1e-10:
                score_without_last = (cumsum[i, idx] - ordered[i, idx]) + penalties_cumsum[idx]
                V[i] = np.clip((q_hat - score_without_last) / ordered[i, idx], 0.0, 1.0)
        sizes = sizes_base - (np.random.random(n_samples) >= V).astype(int)
    else:
        sizes = sizes_base
    
    sizes = np.maximum(sizes, 1)  # Non-empty sets
    
    return [I[i, :sizes[i]] for i in range(n_samples)]


class EntropyStratifiedRAPS_Flexible(nn.Module):
    """
    Entropy-Stratified RAPS with configurable number of strata.
    
    FIXED: Uses randomized calibration scores.
    """
    
    def __init__(self, model, calib_loader, tune_loader, val_loader,
                 alpha=0.1, k_reg=2, n_strata=3, randomized=True, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.n_strata = n_strata
        self.randomized = randomized
        self.num_classes = model.num_classes
        
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [Entropy-RAPS-{n_strata}] Optimizing Lambda with {n_strata} strata...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"   [Entropy-RAPS-{n_strata}] Lambda: {self.lamda_star:.5f}, q_hat: {self.q_hat_star:.4f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute quantile with RANDOMIZED scores."""
        scores = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg, randomized=self.randomized
        )
        return compute_quantile(scores, self.alpha)
    
    def _optimize_lambda_entropy(self):
        """Optimize lambda via minimax over entropy strata."""
        # Define entropy boundaries on tune set
        tune_entropy = calculate_entropy(self.logits_tune)
        quantiles = np.linspace(0, 1, self.n_strata + 1)[1:-1]
        boundaries = np.quantile(tune_entropy, quantiles)
        print(f"      Entropy boundaries: {[f'{b:.3f}' for b in boundaries]}")
        
        # Get strata indices on val set
        val_entropy = calculate_entropy(self.logits_val)
        strata_idxs = []
        
        for i in range(self.n_strata):
            if i == 0:
                mask = val_entropy <= boundaries[0] if len(boundaries) > 0 else np.ones(len(val_entropy), dtype=bool)
            elif i == self.n_strata - 1:
                mask = val_entropy > boundaries[-1]
            else:
                mask = (val_entropy > boundaries[i-1]) & (val_entropy <= boundaries[i])
            strata_idxs.append(np.where(mask)[0])
        
        print(f"      Val strata sizes: {[len(s) for s in strata_idxs]}")
        
        # Lambda selection via minimax
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            probs_val = torch.softmax(self.logits_val, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets(
                probs_val, q_cand, lam, self.k_reg, 
                self.num_classes, randomized=self.randomized
            )
            
            is_covered = np.array([
                self.labels_val[i].item() in pred_sets[i]
                for i in range(len(self.labels_val))
            ])
            
            violations = []
            for s_idx in strata_idxs:
                if len(s_idx) > 0:
                    cov = np.mean(is_covered[s_idx])
                    violations.append(abs(cov - (1 - self.alpha)))
            
            max_viol = max(violations) if violations else 1.0
            avg_size = np.mean([len(s) for s in pred_sets])
            
            if max_viol < min_max_violation:
                min_max_violation = max_viol
                best_lam = lam
                best_q = q_cand
                best_avg_size = avg_size
            elif abs(max_viol - min_max_violation) < 0.005:
                if avg_size < best_avg_size:
                    best_lam = lam
                    best_q = q_cand
                    best_avg_size = avg_size
        
        return best_lam, best_q
    
    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets(
                probs, self.q_hat_star, self.lamda_star, 
                self.k_reg, self.num_classes, randomized=self.randomized
            )
            
            return logits, pred_sets


def main():
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['dataset'] = 'pathmnist'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[1/3] Loading data...")
    data_loader = MedMNISTDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    print("[2/3] Loading trained model...")
    model = get_model(config, device)

    # Try multiple checkpoint paths
    checkpoint_paths = [
        Path('/content/drive/MyDrive/MedConformal_Final_Submission5/pathmnist_final/models/best_model.pth'),
        Path('/content/drive/MyDrive/MedConformal_Final_Submission2/pathmnist_final/models/pathmnist_exp_20260122_153413_best_model.pth'),
    ]
    
    model_loaded = False
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            print(f"   Loading from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("   ✅ Model loaded successfully!")
            model_loaded = True
            break
    
    if not model_loaded:
        print("   ⚠️ Checkpoint not found!")
        return
    
    trained_model = model
    
    # 3-way split
    print("\n[3/3] Running strata ablation...")
    val_ds = val_loader.dataset
    tune_ds, calib_ds, val_split_ds = data_loader.split_validation_3way(
        val_ds, pct_tune=0.3, pct_calib=0.4, pct_val=0.3
    )
    
    tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32, shuffle=False)
    val_split_loader = torch.utils.data.DataLoader(val_split_ds, batch_size=32, shuffle=False)
    
    # Test different n_strata
    results = []
    
    for n_strata in [2, 3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"Testing n_strata = {n_strata}")
        print(f"{'='*60}")
        
        method = EntropyStratifiedRAPS_Flexible(
            trained_model, calib_loader, tune_loader, val_split_loader,
            alpha=0.1, k_reg=2, n_strata=n_strata, randomized=True, device=device
        )
        
        _, _, probs, sets, labels = predict_with_sets(method, test_loader)
        
        # Metrics
        cov = np.mean([labels[i] in sets[i] for i in range(len(labels))])
        size = np.mean([len(s) for s in sets])
        
        # Hard coverage (top stratum)
        ent = entropy(probs, axis=1)
        hard_threshold = np.quantile(ent, 1.0 - 1.0/n_strata)
        hard_mask = ent > hard_threshold
        hard_cov = np.mean([labels[i] in sets[i] for i in range(len(labels)) if hard_mask[i]])
        
        results.append({
            'n_strata': n_strata,
            'lambda': method.lamda_star,
            'q_hat': method.q_hat_star,
            'coverage': f"{cov:.4f}",
            'avg_size': f"{size:.2f}",
            'hard_coverage': f"{hard_cov:.4f}",
        })
        
        print(f"   Lambda: {method.lamda_star:.5f}")
        print(f"   q_hat: {method.q_hat_star:.4f}")
        print(f"   Coverage: {cov:.4f}")
        print(f"   Avg Size: {size:.2f}")
        print(f"   Hard Coverage: {hard_cov:.4f}")
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("STRATA SENSITIVITY RESULTS (FIXED with Randomized Scores)")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Save
    output_path = Path('/content/drive/MyDrive/MedConformal_Final_Submission5/ablation_strata.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")


if __name__ == "__main__":
    main()