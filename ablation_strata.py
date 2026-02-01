import torch
import torch.nn as nn
import yaml
import sys
from pathlib import Path
sys.path.append('.')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model
from models.conformal_wrapper import get_logits_labels, calculate_entropy, _forward_raps, predict_with_sets
import numpy as np
import pandas as pd
from scipy.stats import entropy

class EntropyStratifiedRAPS_Flexible(nn.Module):
    def __init__(self, model, calib_loader, tune_loader, alpha=0.1, 
                 k_reg=2, n_strata=3, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.n_strata = n_strata
        self.num_classes = model.num_classes
        
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        
        print(f"   [Entropy-RAPS-{n_strata}] Optimizing Lambda with {n_strata} strata...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"   [Entropy-RAPS-{n_strata}] Lambda: {self.lamda_star:.5f}")
    
    def _get_qhat(self, logits, labels, lamda):
        probs = torch.softmax(logits, dim=1)
        sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        
        rows = torch.arange(len(labels))
        ranks = torch.zeros(len(labels), dtype=torch.long)
        for i, lbl in enumerate(labels):
            rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
            ranks[i] = rank.item() if len(rank) > 0 else 0
        
        scores = cumsum[rows, ranks] + lamda * torch.clamp(ranks - self.k_reg + 1, min=0)
        n = len(labels)
        return np.quantile(scores.numpy(), np.ceil((n + 1) * (1 - self.alpha)) / n, method='higher')
    
    def _optimize_lambda_entropy(self):
        tune_entropy = calculate_entropy(self.logits_tune)
        
        quantiles = np.linspace(0, 1, self.n_strata + 1)
        boundaries = np.quantile(tune_entropy, quantiles)
        
        strata_idxs = []
        for i in range(self.n_strata):
            if i == 0:
                mask = tune_entropy <= boundaries[i+1]
            elif i == self.n_strata - 1:
                mask = tune_entropy > boundaries[i]
            else:
                mask = (tune_entropy > boundaries[i]) & (tune_entropy <= boundaries[i+1])
            strata_idxs.append(np.where(mask)[0])
        
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q = 0.0, 0.0
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            probs_tune = torch.softmax(self.logits_tune, dim=1)
            sorted_probs, idx = torch.sort(probs_tune, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            rows = torch.arange(len(self.labels_tune))
            ranks = torch.zeros(len(self.labels_tune), dtype=torch.long)
            for i, lbl in enumerate(self.labels_tune):
                rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
                ranks[i] = rank.item() if len(rank) > 0 else 0
            
            scores_tune = cumsum[rows, ranks] + lam * torch.clamp(ranks - self.k_reg + 1, min=0)
            is_covered = (scores_tune <= q_cand).numpy().astype(int)
            
            penalties = lam * torch.clamp(torch.arange(self.num_classes) - self.k_reg + 1, min=0)
            scores_all = cumsum + penalties.unsqueeze(0)
            sizes = (scores_all <= q_cand).sum(dim=1)
            curr_avg_size = sizes.float().mean().item()
            
            violations = []
            for s_idx in strata_idxs:
                if len(s_idx) == 0: continue
                cov = np.mean(is_covered[s_idx])
                violations.append(abs(cov - (1 - self.alpha)))
            max_viol = max(violations) if violations else 1.0
            
            if max_viol < min_max_violation:
                min_max_violation = max_viol
                best_lam = lam
                best_q = q_cand
                best_avg_size = curr_avg_size
            elif abs(max_viol - min_max_violation) < 0.001:
                if curr_avg_size < best_avg_size:
                    min_max_violation = max_viol
                    best_lam = lam
                    best_q = q_cand
                    best_avg_size = curr_avg_size
        
        return best_lam, best_q
    
    def forward(self, x):
        return _forward_raps(self, x)


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

    checkpoint_path = Path('/content/drive/MyDrive/MedConformal_Final_Submission2/pathmnist_final/models/pathmnist_exp_20260122_153413_best_model.pth')
    
    if checkpoint_path.exists():
        print(f"   Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✅ Model loaded successfully!")
    else:
        print("   ⚠️ Checkpoint not found!")
        print(f"   Tried: {checkpoint_path}")
        return
    
    trained_model = model
    
    # Split val
    print("\n[3/3] Running strata ablation...")
    val_ds = val_loader.dataset
    n = len(val_ds)
    n_cal = int(0.7 * n)
    cal_ds, tune_ds = torch.utils.data.random_split(val_ds, [n_cal, n - n_cal])
    
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=32, shuffle=False)
    tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
    
    # Test different n_strata
    results = []
    
    for n_strata in [2, 3, 4, 5]:
        print(f"\n--- Testing n_strata = {n_strata} ---")
        
        method = EntropyStratifiedRAPS_Flexible(
            trained_model, cal_loader, tune_loader,
            alpha=0.1, k_reg=2, n_strata=n_strata, device=device
        )
        
        _, _, probs, sets, labels = predict_with_sets(method, test_loader)
        
        # Metrics
        cov = np.mean([labels[i] in sets[i] for i in range(len(labels))])
        size = np.mean([len(s) for s in sets])
        
        # Hard coverage (top stratum)
        ent = entropy(probs, axis=1)
        hard_threshold = np.quantile(ent, 1.0 - 1.0/n_strata)  # Top stratum
        hard_mask = ent > hard_threshold
        hard_cov = np.mean([labels[i] in sets[i] for i in range(len(labels)) if hard_mask[i]])
        
        results.append({
            'n_strata': n_strata,
            'lambda': method.lamda_star,
            'coverage': f"{cov:.4f}",
            'avg_size': f"{size:.2f}",
            'hard_coverage': f"{hard_cov:.4f}",
        })
        
        print(f"   Lambda: {method.lamda_star:.5f}")
        print(f"   Coverage: {cov:.4f}")
        print(f"   Avg Size: {size:.2f}")
        print(f"   Hard Coverage: {hard_cov:.4f}")
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("STRATA SENSITIVITY RESULTS:")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Save
    output_path = Path('/content/drive/MyDrive/MedConformal_Final_Submission2/ablation_strata.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")

if __name__ == "__main__":
    main()