"""
RAPS Variants with Alternative Difficulty Proxies

This module implements RAPS with margin-based and confidence×trust proxies
as baselines for comparison with entropy-stratified RAPS.

Addresses Stanford Reviewer Point #4 ablation study request.
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
from models.randomization_utils import compute_randomized_quantile


def get_logits_labels(model, loader, device='cuda'):
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


def compute_raps_scores_batch(logits, labels, lamda, k_reg, temperature=1.0):
    """Compute RAPS conformity scores."""
    probs = torch.softmax(logits / temperature, dim=1)
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    rows = torch.arange(len(labels))
    ranks = torch.zeros(len(labels), dtype=torch.long)
    for i, lbl in enumerate(labels):
        rank = (indices[i] == lbl).nonzero(as_tuple=True)[0]
        ranks[i] = rank.item() if len(rank) > 0 else 0
    
    penalty = lamda * torch.clamp(ranks - k_reg + 1, min=0)
    scores = cumsum[rows, ranks] + penalty
    
    return scores.numpy(), sorted_probs.numpy(), indices.numpy()


# ============================================================================
# MARGIN-STRATIFIED RAPS
# ============================================================================

class MarginStratifiedRAPS(nn.Module):
    """
    RAPS with margin-based stratification.
    
    Baseline comparison to test if margin-based difficulty proxy works as well
    as entropy for worst-case coverage optimization.
    
    Protocol (same as EntropyRAPS):
    1. Tune set: Define margin tertile boundaries
    2. Calib set: Compute q_hat(λ) for each λ
    3. Val set: Select λ via minimax over margin strata
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
        self.lamda_star, self.q_hat_star, self.U = self._optimize_lambda_margin()
        print(f"   [Margin-RAPS] Lambda: {self.lamda_star:.5f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute quantile with randomization."""
        scores, _, _ = compute_raps_scores_batch(
            logits, labels, lamda, self.k_reg, temperature=1.0
        )
        q_hat, U = compute_randomized_quantile(scores, self.alpha, self.randomized)
        return q_hat, U
    
    def _optimize_lambda_margin(self):
        """
        Select λ via minimax over margin-based strata.
        
        STEP 1: Define margin boundaries on TUNE set
        STEP 2: For each λ, compute q_hat(λ) on CALIB set
        STEP 3: Evaluate stratified coverage on VAL set
        STEP 4: Select λ* via minimax
        """
        # STEP 1: Margin boundaries on tune set
        tune_margin = compute_margin_proxy(self.logits_tune)
        th1, th2 = np.quantile(tune_margin, [0.33, 0.66])
        
        print(f"      Margin boundaries (from tune set): [{th1:.3f}, {th2:.3f}]")
        
        # Apply to validation set
        val_margin = compute_margin_proxy(self.logits_val)
        strata_idxs = get_stratum_indices(val_margin, [th1, th2])
        
        print(f"      Val set strata: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # STEP 2-4: Lambda selection
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q, best_U = 0.0, 0.0, None
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand, U_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on validation set
            probs_val = torch.softmax(self.logits_val, dim=1)
            sorted_probs, idx = torch.sort(probs_val, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            rows = torch.arange(len(self.labels_val))
            ranks = torch.zeros(len(self.labels_val), dtype=torch.long)
            for i, lbl in enumerate(self.labels_val):
                rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
                ranks[i] = rank.item() if len(rank) > 0 else 0
            
            scores_val = cumsum[rows, ranks] + lam * torch.clamp(ranks - self.k_reg + 1, min=0)
            is_covered = (scores_val <= q_cand).numpy().astype(int)
            
            # Compute average size for tie-breaking
            penalties = lam * torch.clamp(torch.arange(self.num_classes) - self.k_reg + 1, min=0)
            scores_all = cumsum + penalties.unsqueeze(0)
            sizes = (scores_all <= q_cand).sum(dim=1)
            curr_avg_size = sizes.float().mean().item()
            
            # Minimax selection
            violations = []
            for s_idx in strata_idxs:
                if len(s_idx) == 0: continue
                cov = np.mean(is_covered[s_idx])
                violations.append(abs(cov - (1 - self.alpha)))
            max_viol = max(violations) if violations else 1.0
            
            if max_viol < min_max_violation:
                min_max_violation = max_viol
                best_lam, best_q, best_U = lam, q_cand, U_cand
                best_avg_size = curr_avg_size
            elif abs(max_viol - min_max_violation) < 0.001:
                if curr_avg_size < best_avg_size:
                    min_max_violation = max_viol
                    best_lam, best_q, best_U = lam, q_cand, U_cand
                    best_avg_size = curr_avg_size
        
        return best_lam, best_q, best_U
    
    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            penalties = self.lamda_star * torch.clamp(
                torch.arange(self.num_classes) - self.k_reg + 1, min=0
            )
            
            for i in range(x.size(0)):
                scores = cumsum[i] + penalties
                
                if not self.randomized or self.U is None:
                    mask = scores <= self.q_hat_star
                    k_star = mask.nonzero()[-1].item() if mask.sum() > 0 else 0
                else:
                    below = scores < self.q_hat_star
                    k_base = below.nonzero()[-1].item() + 1 if below.sum() > 0 else 0
                    
                    if k_base < len(scores):
                        at_boundary = torch.isclose(
                            scores[k_base], torch.tensor(self.q_hat_star), atol=1e-6
                        )
                        if at_boundary and np.random.rand() < self.U:
                            k_star = k_base
                        else:
                            k_star = k_base - 1 if k_base > 0 else 0
                    else:
                        k_star = k_base - 1
                    
                    k_star = max(0, k_star)
                
                pred_set = indices[i, :k_star + 1].cpu().numpy()
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets


# ============================================================================
# CONFIDENCE × TRUST STRATIFIED RAPS
# ============================================================================

class ConfTrustStratifiedRAPS(nn.Module):
    """
    RAPS with confidence×trust stratification.
    
    Baseline comparison testing the conf×trust product as difficulty proxy.
    
    Protocol (same as EntropyRAPS):
    1. Tune set: Define conf×trust tertile boundaries
    2. Calib set: Compute q_hat(λ) for each λ
    3. Val set: Select λ via minimax over conf×trust strata
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
        self.lamda_star, self.q_hat_star, self.U = self._optimize_lambda_conftrust()
        print(f"   [ConfTrust-RAPS] Lambda: {self.lamda_star:.5f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute quantile with randomization."""
        scores, _, _ = compute_raps_scores_batch(
            logits, labels, lamda, self.k_reg, temperature=1.0
        )
        q_hat, U = compute_randomized_quantile(scores, self.alpha, self.randomized)
        return q_hat, U
    
    def _optimize_lambda_conftrust(self):
        """
        Select λ via minimax over confidence×trust strata.
        
        STEP 1: Define conf×trust boundaries on TUNE set
        STEP 2: For each λ, compute q_hat(λ) on CALIB set
        STEP 3: Evaluate stratified coverage on VAL set
        STEP 4: Select λ* via minimax
        """
        # STEP 1: ConfTrust boundaries on tune set
        tune_conftrust = compute_confidence_trust_proxy(self.logits_tune)
        th1, th2 = np.quantile(tune_conftrust, [0.33, 0.66])
        
        print(f"      ConfTrust boundaries (from tune set): [{th1:.3f}, {th2:.3f}]")
        
        # Apply to validation set
        val_conftrust = compute_confidence_trust_proxy(self.logits_val)
        strata_idxs = get_stratum_indices(val_conftrust, [th1, th2])
        
        print(f"      Val set strata: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # STEP 2-4: Lambda selection
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q, best_U = 0.0, 0.0, None
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand, U_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on validation set
            probs_val = torch.softmax(self.logits_val, dim=1)
            sorted_probs, idx = torch.sort(probs_val, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            rows = torch.arange(len(self.labels_val))
            ranks = torch.zeros(len(self.labels_val), dtype=torch.long)
            for i, lbl in enumerate(self.labels_val):
                rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
                ranks[i] = rank.item() if len(rank) > 0 else 0
            
            scores_val = cumsum[rows, ranks] + lam * torch.clamp(ranks - self.k_reg + 1, min=0)
            is_covered = (scores_val <= q_cand).numpy().astype(int)
            
            # Compute average size
            penalties = lam * torch.clamp(torch.arange(self.num_classes) - self.k_reg + 1, min=0)
            scores_all = cumsum + penalties.unsqueeze(0)
            sizes = (scores_all <= q_cand).sum(dim=1)
            curr_avg_size = sizes.float().mean().item()
            
            # Minimax selection
            violations = []
            for s_idx in strata_idxs:
                if len(s_idx) == 0: continue
                cov = np.mean(is_covered[s_idx])
                violations.append(abs(cov - (1 - self.alpha)))
            max_viol = max(violations) if violations else 1.0
            
            if max_viol < min_max_violation:
                min_max_violation = max_viol
                best_lam, best_q, best_U = lam, q_cand, U_cand
                best_avg_size = curr_avg_size
            elif abs(max_viol - min_max_violation) < 0.001:
                if curr_avg_size < best_avg_size:
                    min_max_violation = max_viol
                    best_lam, best_q, best_U = lam, q_cand, U_cand
                    best_avg_size = curr_avg_size
        
        return best_lam, best_q, best_U
    
    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            penalties = self.lamda_star * torch.clamp(
                torch.arange(self.num_classes) - self.k_reg + 1, min=0
            )
            
            for i in range(x.size(0)):
                scores = cumsum[i] + penalties
                
                if not self.randomized or self.U is None:
                    mask = scores <= self.q_hat_star
                    k_star = mask.nonzero()[-1].item() if mask.sum() > 0 else 0
                else:
                    below = scores < self.q_hat_star
                    k_base = below.nonzero()[-1].item() + 1 if below.sum() > 0 else 0
                    
                    if k_base < len(scores):
                        at_boundary = torch.isclose(
                            scores[k_base], torch.tensor(self.q_hat_star), atol=1e-6
                        )
                        if at_boundary and np.random.rand() < self.U:
                            k_star = k_base
                        else:
                            k_star = k_base - 1 if k_base > 0 else 0
                    else:
                        k_star = k_base - 1
                    
                    k_star = max(0, k_star)
                
                pred_set = indices[i, :k_star + 1].cpu().numpy()
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets
