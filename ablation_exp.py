import torch
import yaml
import sys
from pathlib import Path
sys.path.append('.')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model
from models.conformal_wrapper import EntropyStratifiedRAPS_Flexible, predict_with_sets
from training.trainer import Trainer
import numpy as np
import pandas as pd

class EntropyStratifiedRAPS_Flexible(nn.Module):
    """
    EntropyRAPS with flexible number of strata.
    
    This is a modified version of your EntropyStratifiedRAPS that allows
    experimenting with different n_strata values.
    """
    
    def __init__(self, model, calib_loader, tune_loader, alpha=0.1, 
                 k_reg=2, n_strata=3, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.n_strata = n_strata
        self.num_classes = model.num_classes
        
        # Get logits
        from models.conformal_wrapper import get_logits_labels, calculate_entropy
        
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        
        print(f"   [Entropy-RAPS-{n_strata}] Optimizing Lambda with {n_strata} strata...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"   [Entropy-RAPS-{n_strata}] Lambda: {self.lamda_star:.5f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Same as your implementation."""
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
        """Modified to handle flexible n_strata."""
        from models.conformal_wrapper import calculate_entropy
        
        tune_entropy = calculate_entropy(self.logits_tune)
        
        # Compute quantiles for n_strata
        quantiles = np.linspace(0, 1, self.n_strata + 1)
        boundaries = np.quantile(tune_entropy, quantiles)
        
        # Assign to strata
        strata_idxs = []
        for i in range(self.n_strata):
            if i == 0:
                mask = tune_entropy <= boundaries[i+1]
            elif i == self.n_strata - 1:
                mask = tune_entropy > boundaries[i]
            else:
                mask = (tune_entropy > boundaries[i]) & (tune_entropy <= boundaries[i+1])
            strata_idxs.append(np.where(mask)[0])
        
        # Rest is same as original
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
        """Same as your implementation."""
        from models.conformal_wrapper import _forward_raps
        return _forward_raps(self, x)


def run_strata_sensitivity(model, calib_loader, tune_loader, test_loader, 
                          strata_counts=[2, 3, 4, 5], device='cuda'):
    """
    Run sensitivity analysis on number of strata.
    
    Args:
        model: Trained model
        calib_loader, tune_loader, test_loader: Data loaders
        strata_counts: List of n_strata values to test
        device: 'cuda' or 'cpu'
    
    Returns:
        DataFrame with results
    """
    from models.conformal_wrapper import predict_with_sets
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: Number of Strata")
    print("="*70)
    
    results = []
    
    for n_strata in strata_counts:
        print(f"\n--- Testing n_strata = {n_strata} ---")
        
        # Create method
        method = EntropyStratifiedRAPS_Flexible(
            model, calib_loader, tune_loader,
            alpha=0.1, k_reg=2, n_strata=n_strata, device=device
        )
        
        # Evaluate
        logits, preds, probs, sets, labels = predict_with_sets(method, test_loader)
        
        # Compute metrics
        coverage = np.mean([labels[i] in sets[i] for i in range(len(labels))])
        avg_size = np.mean([len(s) for s in sets])
        
        # Compute hard-case coverage (using n_strata-based stratification)
        ent = entropy(probs, axis=1)
        quantiles = np.linspace(0, 1, n_strata + 1)
        boundaries = np.quantile(ent, quantiles)
        hard_mask = ent > boundaries[-2]  # Last stratum
        hard_cov = np.mean([labels[i] in sets[i] for i in range(len(labels)) if hard_mask[i]])
        
        results.append({
            'n_strata': n_strata,
            'lambda': method.lamda_star,
            'coverage': coverage,
            'avg_size': avg_size,
            'hard_coverage': hard_cov,
        })
        
        print(f"   Lambda: {method.lamda_star:.5f}")
        print(f"   Coverage: {coverage:.4f}")
        print(f"   Avg Size: {avg_size:.2f}")
        print(f"   Hard Coverage: {hard_cov:.4f}")
    
    df = pd.DataFrame(results)
    return df


# ============================================================================
# ALTERNATIVE PROXY ABLATION
# ============================================================================

def compute_alternative_proxies(logits: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute alternative difficulty proxies.
    
    Args:
        logits: Tensor of shape [n, num_classes]
    
    Returns:
        Dictionary with proxy name -> proxy values
    """
    probs = torch.softmax(logits, dim=1)
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    
    proxies = {}
    
    # 1. Entropy (original)
    proxies['entropy'] = entropy(probs.numpy(), axis=1)
    
    # 2. Max Probability (inverse)
    proxies['max_prob'] = (1 - sorted_probs[:, 0]).numpy()
    
    # 3. Top-1 Margin
    if probs.shape[1] >= 2:
        proxies['margin'] = (sorted_probs[:, 0] - sorted_probs[:, 1]).numpy()
        proxies['margin'] = -proxies['margin']  # Negative so higher = harder
    
    # 4. Temperature-scaled entropy (T=2.0)
    probs_scaled = torch.softmax(logits / 2.0, dim=1)
    proxies['temp_entropy'] = entropy(probs_scaled.numpy(), axis=1)
    
    return proxies


class ProxyStratifiedRAPS(nn.Module):
    """
    RAPS with minimax tuning using alternative proxy.
    """
    
    def __init__(self, model, calib_loader, tune_loader, alpha=0.1,
                 k_reg=2, proxy_type='entropy', n_strata=3, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.n_strata = n_strata
        self.proxy_type = proxy_type
        self.num_classes = model.num_classes
        
        from models.conformal_wrapper import get_logits_labels
        
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        
        print(f"   [RAPS-{proxy_type}] Optimizing Lambda with {proxy_type} stratification...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_proxy()
        print(f"   [RAPS-{proxy_type}] Lambda: {self.lamda_star:.5f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Same as before."""
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
    
    def _optimize_lambda_proxy(self):
        """Optimize using alternative proxy."""
        # Compute proxy values
        tune_proxies = compute_alternative_proxies(self.logits_tune)
        proxy_values = tune_proxies[self.proxy_type]
        
        # Define strata
        quantiles = np.linspace(0, 1, self.n_strata + 1)
        boundaries = np.quantile(proxy_values, quantiles)
        
        strata_idxs = []
        for i in range(self.n_strata):
            if i == 0:
                mask = proxy_values <= boundaries[i+1]
            elif i == self.n_strata - 1:
                mask = proxy_values > boundaries[i]
            else:
                mask = (proxy_values > boundaries[i]) & (proxy_values <= boundaries[i+1])
            strata_idxs.append(np.where(mask)[0])
        
        # Optimize lambda (rest is same)
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
        from models.conformal_wrapper import _forward_raps
        return _forward_raps(self, x)


def main():
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['dataset'] = 'pathmnist'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load & train
    data_loader = MedMNISTDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    print("Training model...")
    model = get_model(config, device)
    trainer = Trainer(model, config, device)
    trained_model = trainer.train(train_loader, val_loader, None)
    
    # Split val
    val_ds = val_loader.dataset
    n = len(val_ds)
    n_cal = int(0.7 * n)
    cal_ds, tune_ds = torch.utils.data.random_split(val_ds, [n_cal, n - n_cal])
    
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=32, shuffle=False)
    tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
    
    # Test different n_strata
    print("\nTesting different number of strata...")
    results = []
    
    for n_strata in [2, 3, 4, 5]:
        print(f"  n_strata={n_strata}...")
        method = EntropyStratifiedRAPS_Flexible(
            trained_model, cal_loader, tune_loader,
            alpha=0.1, k_reg=2, n_strata=n_strata, device=device
        )
        
        _, _, _, sets, labels = predict_with_sets(method, test_loader)
        
        cov = np.mean([labels[i] in sets[i] for i in range(len(labels))])
        size = np.mean([len(s) for s in sets])
        
        results.append({
            'n_strata': n_strata,
            'lambda': method.lamda_star,
            'coverage': cov,
            'avg_size': size
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("ABLATION RESULTS:")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv('ablation_strata.csv', index=False)
    print("\n✅ Saved to ablation_strata.csv")

if __name__ == "__main__":
    main()