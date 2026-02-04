import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('.')
from models.alternative_proxies import (
    compute_margin_proxy,
    compute_confidence_trust_proxy,
    get_stratum_indices
)


def get_logits_labels(model, loader, device='cuda'):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            logits_list.append(out.cpu())
            if len(y.shape) > 1:
                y = y.squeeze()
            labels_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def compute_raps_scores_randomized(logits, labels, lamda, k_reg, temperature=1.0, randomized=True):
    """Compute RAPS scores with randomization (matching original paper)."""
    probs = torch.softmax(logits / temperature, dim=1)
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    n_samples = len(labels)
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        rank_tensor = (indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        rank = rank_tensor.item() if len(rank_tensor) > 0 else 0
        p_y = sorted_probs[i, rank].item()
        penalty = lamda * max(rank - k_reg + 1, 0)
        U = np.random.random() if randomized else 1.0
        
        if rank == 0:
            scores[i] = U * p_y + penalty
        else:
            scores[i] = U * p_y + cumsum[i, rank - 1].item() + penalty
    
    return scores


def compute_quantile(scores, alpha):
    return np.quantile(scores, 1 - alpha, method='higher')


def construct_prediction_sets_gcq(probs, q_hat, lamda=0.0, k_reg=2,
                                   randomized=True, allow_zero_sets=False):
    """Construct prediction sets using gcq (matching original paper)."""
    n_samples, n_classes = probs.shape
    
    I = probs.argsort(axis=1)[:, ::-1]
    ordered = np.sort(probs, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    
    penalties = np.zeros(n_classes)
    penalties[k_reg:] = lamda
    penalties_cumsum = np.cumsum(penalties)
    
    sizes_base = ((cumsum + penalties_cumsum) <= q_hat).sum(axis=1) + 1
    sizes_base = np.minimum(sizes_base, n_classes)
    
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
    
    if q_hat >= 1.0:
        sizes[:] = n_classes
    
    if not allow_zero_sets:
        sizes = np.maximum(sizes, 1)
    
    return [I[i, :sizes[i]] for i in range(n_samples)]


class MarginStratifiedRAPS(nn.Module):
    """RAPS with margin-based stratification."""
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, randomized=True, allow_zero_sets=False, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = model.num_classes
        
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [Margin-RAPS] Optimizing (allow_zero_sets={allow_zero_sets})...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda()
        print(f"   [Margin-RAPS] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")

    def _get_qhat(self, logits, labels, lamda):
        scores = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg, 1.0, self.randomized)
        return compute_quantile(scores, self.alpha)

    def _optimize_lambda(self):
        tune_margin = compute_margin_proxy(self.logits_tune)
        th1, th2 = np.quantile(tune_margin, [0.33, 0.66])
        print(f"      Margin boundaries: [{th1:.3f}, {th2:.3f}]")
        
        val_margin = compute_margin_proxy(self.logits_val)
        strata_idxs = get_stratum_indices(val_margin, [th1, th2])
        print(f"      Val strata: Easy={len(strata_idxs[0])}, Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_max_viol, best_size = float('inf'), float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            probs_val = torch.softmax(self.logits_val, dim=1).cpu().numpy()
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                self.randomized, self.allow_zero_sets)
            
            is_covered = np.array([self.labels_val[i].item() in pred_sets[i]
                                   for i in range(len(self.labels_val))])
            
            violations = [abs(np.mean(is_covered[idx]) - (1 - self.alpha))
                         for idx in strata_idxs if len(idx) > 0]
            max_viol = max(violations) if violations else 1.0
            avg_size = np.mean([len(s) for s in pred_sets])
            
            if max_viol < min_max_viol:
                min_max_viol, best_lam, best_q, best_size = max_viol, lam, q_cand, avg_size
            elif abs(max_viol - min_max_viol) < 0.005 and avg_size < best_size:
                best_lam, best_q, best_size = lam, q_cand, avg_size
        
        return best_lam, best_q

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star, self.lamda_star, self.k_reg,
                self.randomized, self.allow_zero_sets)
            return logits, pred_sets


class ConfTrustStratifiedRAPS(nn.Module):
    """RAPS with confidence×trust stratification."""
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, randomized=True, allow_zero_sets=False, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = model.num_classes
        
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [ConfTrust-RAPS] Optimizing (allow_zero_sets={allow_zero_sets})...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda()
        print(f"   [ConfTrust-RAPS] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")

    def _get_qhat(self, logits, labels, lamda):
        scores = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg, 1.0, self.randomized)
        return compute_quantile(scores, self.alpha)

    def _optimize_lambda(self):
        tune_ct = compute_confidence_trust_proxy(self.logits_tune)
        th1, th2 = np.quantile(tune_ct, [0.33, 0.66])
        print(f"      ConfTrust boundaries: [{th1:.3f}, {th2:.3f}]")
        
        val_ct = compute_confidence_trust_proxy(self.logits_val)
        strata_idxs = get_stratum_indices(val_ct, [th1, th2])
        print(f"      Val strata: Easy={len(strata_idxs[0])}, Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_max_viol, best_size = float('inf'), float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            probs_val = torch.softmax(self.logits_val, dim=1).cpu().numpy()
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                self.randomized, self.allow_zero_sets)
            
            is_covered = np.array([self.labels_val[i].item() in pred_sets[i]
                                   for i in range(len(self.labels_val))])
            
            violations = [abs(np.mean(is_covered[idx]) - (1 - self.alpha))
                         for idx in strata_idxs if len(idx) > 0]
            max_viol = max(violations) if violations else 1.0
            avg_size = np.mean([len(s) for s in pred_sets])
            
            if max_viol < min_max_viol:
                min_max_viol, best_lam, best_q, best_size = max_viol, lam, q_cand, avg_size
            elif abs(max_viol - min_max_viol) < 0.005 and avg_size < best_size:
                best_lam, best_q, best_size = lam, q_cand, avg_size
        
        return best_lam, best_q

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star, self.lamda_star, self.k_reg,
                self.randomized, self.allow_zero_sets)
            return logits, pred_sets