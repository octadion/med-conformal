"""
RAPS Variants with Alternative Difficulty Proxies (FIXED)

This module implements RAPS with margin-based and confidence×trust proxies
as baselines for comparison with entropy-stratified RAPS.

Addresses Stanford Reviewer Point #4 ablation study request.

FIXED: Uses randomized calibration scores matching original RAPS paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# Import helper functions
import sys
sys.path.append('.')
from models.alternative_proxies import (
    compute_margin_proxy,
    compute_confidence_trust_proxy,
    get_stratum_indices
)


def get_logits_labels(model, loader, device='cuda'):
    """Extract logits and labels from dataloader."""
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            logits_list.append(out.cpu())
            if len(y.shape) > 1:
                y = y.squeeze()
            labels_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def compute_raps_scores_randomized(logits: torch.Tensor, 
                                    labels: torch.Tensor, 
                                    lamda: float,
                                    k_reg: int,
                                    temperature: float = 1.0,
                                    randomized: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RAPS conformity scores WITH randomization (matching original paper).
    
    FIXED: From original conformal.py get_tau():
        score = U * p_y + cumsum[rank-1] + penalty
    
    Args:
        logits: Model outputs (N, K)
        labels: True labels (N,)
        lamda: RAPS regularization parameter
        k_reg: Regularization starting point
        temperature: Temperature scaling factor
        randomized: If True, use randomized scores (recommended)
    
    Returns:
        Tuple of (scores, sorted_probs, indices)
    """
    probs = torch.softmax(logits / temperature, dim=1)
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    n_samples = len(labels)
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Find rank of true label in sorted order
        rank_tensor = (indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        rank = rank_tensor.item() if len(rank_tensor) > 0 else 0
        
        p_y = sorted_probs[i, rank].item()
        
        # Compute penalty: λ * max(rank - k_reg + 1, 0)
        penalty = lamda * max(rank - k_reg + 1, 0)
        
        if randomized:
            U = np.random.random()
        else:
            U = 1.0
        
        if rank == 0:
            # True label is top-1
            scores[i] = U * p_y + penalty
        else:
            # True label is rank k > 0
            cumsum_before = cumsum[i, rank - 1].item()
            scores[i] = U * p_y + cumsum_before + penalty
    
    return scores, sorted_probs.numpy(), indices.numpy()


def compute_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute conformal quantile (matching original paper)."""
    return np.quantile(scores, 1 - alpha, method='higher')


def construct_prediction_sets_gcq(probs: np.ndarray, 
                                   q_hat: float,
                                   lamda: float = 0.0,
                                   k_reg: int = 2,
                                   randomized: bool = True,
                                   allow_zero_sets: bool = False) -> List[np.ndarray]:
    """
    Construct prediction sets using Generalized Conditional Quantile (gcq).
    
    This is the EXACT algorithm from original conformal.py.
    """
    n_samples, n_classes = probs.shape
    
    # Sort probabilities descending
    I = probs.argsort(axis=1)[:, ::-1]
    ordered = np.sort(probs, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    
    # Compute cumulative penalties (flat penalty after k_reg, like original)
    penalties = np.zeros(n_classes)
    penalties[k_reg:] = lamda
    penalties_cumsum = np.cumsum(penalties)
    
    # Base set sizes
    sizes_base = ((cumsum + penalties_cumsum) <= q_hat).sum(axis=1) + 1
    sizes_base = np.minimum(sizes_base, n_classes)
    
    if randomized:
        # Compute V for randomized inclusion
        V = np.zeros(n_samples)
        for i in range(n_samples):
            idx = sizes_base[i] - 1
            if idx >= 0 and ordered[i, idx] > 1e-10:
                score_without_last = (cumsum[i, idx] - ordered[i, idx]) + penalties_cumsum[idx]
                V[i] = (q_hat - score_without_last) / ordered[i, idx]
                V[i] = np.clip(V[i], 0.0, 1.0)
        
        # Randomized exclusion
        sizes = sizes_base - (np.random.random(n_samples) >= V).astype(int)
    else:
        sizes = sizes_base
    
    # Handle special cases
    if q_hat >= 1.0:
        sizes[:] = n_classes
    
    if not allow_zero_sets:
        sizes = np.maximum(sizes, 1)
    
    # Build prediction sets
    prediction_sets = [I[i, :sizes[i]] for i in range(n_samples)]
    
    return prediction_sets


# ============================================================================
# MARGIN-STRATIFIED RAPS (FIXED)
# ============================================================================

class MarginStratifiedRAPS(nn.Module):
    """
    RAPS with margin-based stratification (FIXED).
    
    FIXED: Uses randomized calibration scores matching original RAPS paper.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, randomized=True, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.num_classes = model.num_classes
        
        # Extract logits
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [Margin-RAPS] Optimizing Lambda (randomized={randomized})...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_margin()
        print(f"   [Margin-RAPS] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute quantile on calibration set (FIXED with randomization)."""
        scores, _, _ = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg, 
            temperature=1.0, randomized=self.randomized
        )
        return compute_quantile(scores, self.alpha)
    
    def _optimize_lambda_margin(self):
        """Select λ via minimax over margin-based strata."""
        # STEP 1: Margin boundaries on tune set
        tune_margin = compute_margin_proxy(self.logits_tune)
        th1, th2 = np.quantile(tune_margin, [0.33, 0.66])
        
        print(f"      Margin boundaries: [{th1:.3f}, {th2:.3f}]")
        
        # Apply to validation set
        val_margin = compute_margin_proxy(self.logits_val)
        strata_idxs = get_stratum_indices(val_margin, [th1, th2])
        
        print(f"      Val strata: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # Lambda selection
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on validation set
            probs_val = torch.softmax(self.logits_val, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            # Compute coverage per stratum
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
            
            # Minimax selection
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
        """Forward pass with FIXED prediction set construction."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star,
                lamda=self.lamda_star, k_reg=self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            return logits, prediction_sets


# ============================================================================
# CONFIDENCE × TRUST STRATIFIED RAPS (FIXED)
# ============================================================================

class ConfTrustStratifiedRAPS(nn.Module):
    """
    RAPS with confidence×trust stratification (FIXED).
    
    FIXED: Uses randomized calibration scores matching original RAPS paper.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, randomized=True, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.num_classes = model.num_classes
        
        # Extract logits
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [ConfTrust-RAPS] Optimizing Lambda (randomized={randomized})...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_conftrust()
        print(f"   [ConfTrust-RAPS] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute quantile on calibration set (FIXED with randomization)."""
        scores, _, _ = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg,
            temperature=1.0, randomized=self.randomized
        )
        return compute_quantile(scores, self.alpha)
    
    def _optimize_lambda_conftrust(self):
        """Select λ via minimax over confidence×trust strata."""
        # STEP 1: ConfTrust boundaries on tune set
        tune_conftrust = compute_confidence_trust_proxy(self.logits_tune)
        th1, th2 = np.quantile(tune_conftrust, [0.33, 0.66])
        
        print(f"      ConfTrust boundaries: [{th1:.3f}, {th2:.3f}]")
        
        # Apply to validation set
        val_conftrust = compute_confidence_trust_proxy(self.logits_val)
        strata_idxs = get_stratum_indices(val_conftrust, [th1, th2])
        
        print(f"      Val strata: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # Lambda selection
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on validation set
            probs_val = torch.softmax(self.logits_val, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            # Compute coverage per stratum
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
            
            # Minimax selection
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
        """Forward pass with FIXED prediction set construction."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star,
                lamda=self.lamda_star, k_reg=self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            return logits, prediction_sets