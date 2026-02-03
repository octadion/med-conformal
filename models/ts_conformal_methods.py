"""
Temperature-Scaled Conformal Prediction Methods (FIXED)

This module provides temperature-scaled variants of conformal prediction methods.
Temperature scaling improves probability calibration before computing prediction sets.

Baseline comparison requested by Stanford reviewer (Point #3).

FIXED: Uses randomized calibration scores matching original RAPS paper.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from typing import List, Tuple

# Import base temperature scaling
import sys
sys.path.append('.')
from models.temperature_scaling import TemperatureScaler


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


def calculate_entropy(logits, temperature=1.0):
    """Calculate entropy with optional temperature scaling."""
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    return entropy(probs, axis=1)


# =============================================================================
# FIXED SCORE COMPUTATION WITH RANDOMIZATION
# =============================================================================

def compute_aps_scores_randomized(logits: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   temperature: float = 1.0,
                                   randomized: bool = True) -> np.ndarray:
    """
    Compute APS conformity scores WITH randomization (matching original paper).
    
    FIXED: score = U * p_y + cumsum[rank-1] instead of cumsum[rank]
    """
    probs = torch.softmax(logits / temperature, dim=1)
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    n_samples = len(labels)
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        rank_tensor = (indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        rank = rank_tensor.item() if len(rank_tensor) > 0 else 0
        
        p_y = sorted_probs[i, rank].item()
        U = np.random.random() if randomized else 1.0
        
        if rank == 0:
            scores[i] = U * p_y
        else:
            cumsum_before = cumsum[i, rank - 1].item()
            scores[i] = U * p_y + cumsum_before
    
    return scores


def compute_raps_scores_randomized(logits: torch.Tensor, 
                                    labels: torch.Tensor,
                                    lamda: float,
                                    k_reg: int,
                                    temperature: float = 1.0,
                                    randomized: bool = True) -> np.ndarray:
    """
    Compute RAPS conformity scores WITH randomization (matching original paper).
    
    FIXED: score = U * p_y + cumsum[rank-1] + penalty
    """
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
            cumsum_before = cumsum[i, rank - 1].item()
            scores[i] = U * p_y + cumsum_before + penalty
    
    return scores


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
    EXACT algorithm from original conformal.py.
    """
    n_samples, n_classes = probs.shape
    
    I = probs.argsort(axis=1)[:, ::-1]
    ordered = np.sort(probs, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    
    # Flat penalty after k_reg (like original)
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
                V[i] = (q_hat - score_without_last) / ordered[i, idx]
                V[i] = np.clip(V[i], 0.0, 1.0)
        
        sizes = sizes_base - (np.random.random(n_samples) >= V).astype(int)
    else:
        sizes = sizes_base
    
    if q_hat >= 1.0:
        sizes[:] = n_classes
    
    if not allow_zero_sets:
        sizes = np.maximum(sizes, 1)
    
    return [I[i, :sizes[i]] for i in range(n_samples)]


# ============================================================================
# TEMPERATURE-SCALED APS (FIXED)
# ============================================================================

class TemperatureScaledAPS(nn.Module):
    """
    APS with temperature scaling (FIXED).
    
    FIXED: Uses randomized calibration scores matching original paper.
    """
    
    def __init__(self, model, calib_loader, alpha=0.1, 
                 ts_max_iters=50, randomized=True, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.device = device
        
        print("   [TS-APS] Step 1: Calibrating temperature...")
        
        # Calibrate temperature
        self.temp_scaler = TemperatureScaler(
            model, calib_loader, device=device,
            max_iters=ts_max_iters, verbose=False
        )
        self.temperature = self.temp_scaler.temperature
        
        print(f"   [TS-APS] Optimal T = {self.temperature:.4f}")
        print("   [TS-APS] Step 2: Computing APS quantile...")
        
        # FIXED: Use randomized calibration scores
        logits, labels = get_logits_labels(model, calib_loader, device)
        scores = compute_aps_scores_randomized(
            logits, labels, self.temperature, randomized=randomized
        )
        
        self.q_hat = compute_quantile(scores, alpha)
        
        print(f"   [TS-APS] Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"   [TS-APS] Q_hat: {self.q_hat:.4f}")
    
    def forward(self, x):
        """Forward pass with FIXED prediction set construction."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat,
                lamda=0.0, k_reg=2,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            return logits, prediction_sets


# ============================================================================
# TEMPERATURE-SCALED RAPS (Size Optimized) (FIXED)
# ============================================================================

class TemperatureScaledRAPS(nn.Module):
    """
    Size-optimized RAPS with temperature scaling (FIXED).
    
    FIXED: Uses randomized calibration scores matching original paper.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, ts_max_iters=50, randomized=True, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.device = device
        self.num_classes = model.num_classes
        
        print("   [TS-RAPS] Step 1: Calibrating temperature...")
        
        self.temp_scaler = TemperatureScaler(
            model, calib_loader, device=device,
            max_iters=ts_max_iters, verbose=False
        )
        self.temperature = self.temp_scaler.temperature
        
        print(f"   [TS-RAPS] Optimal T = {self.temperature:.4f}")
        
        # Extract logits
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print("   [TS-RAPS] Step 2: Optimizing lambda...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_size()
        print(f"   [TS-RAPS] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute RAPS quantile (FIXED with randomization)."""
        scores = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg,
            self.temperature, randomized=self.randomized
        )
        return compute_quantile(scores, self.alpha)
    
    def _optimize_lambda_size(self):
        """Select lambda for minimum set size."""
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_size = float('inf')
        target_cov = 1 - self.alpha
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on val set
            probs_val = torch.softmax(self.logits_val / self.temperature, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            coverage = np.mean([
                self.labels_val[i].item() in pred_sets[i]
                for i in range(len(self.labels_val))
            ])
            avg_size = np.mean([len(s) for s in pred_sets])
            
            if coverage >= target_cov - 0.02:
                if avg_size < min_size:
                    min_size = avg_size
                    best_lam = lam
                    best_q = q_cand
        
        if best_q == 0.0:
            best_lam = 0.0
            best_q = self._get_qhat(self.logits_cal, self.labels_cal, 0.0)
        
        return best_lam, best_q
    
    def forward(self, x):
        """Forward pass with FIXED prediction set construction."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star,
                lamda=self.lamda_star, k_reg=self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            return logits, prediction_sets


# ============================================================================
# TEMPERATURE-SCALED ENTROPY-RAPS (Ours + TS) (FIXED)
# ============================================================================

class TemperatureScaledEntropyRAPS(nn.Module):
    """
    Entropy-stratified RAPS with temperature scaling (FIXED).
    
    FIXED: Uses randomized calibration scores matching original paper.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, ts_max_iters=50, randomized=True, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.device = device
        self.num_classes = model.num_classes
        
        print("   [TS-EntropyRAPS] Step 1: Calibrating temperature...")
        
        self.temp_scaler = TemperatureScaler(
            model, calib_loader, device=device,
            max_iters=ts_max_iters, verbose=False
        )
        self.temperature = self.temp_scaler.temperature
        
        print(f"   [TS-EntropyRAPS] Optimal T = {self.temperature:.4f}")
        
        # Extract logits
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print("   [TS-EntropyRAPS] Step 2: Optimizing lambda via minimax...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"   [TS-EntropyRAPS] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute RAPS quantile (FIXED with randomization)."""
        scores = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg,
            self.temperature, randomized=self.randomized
        )
        return compute_quantile(scores, self.alpha)
    
    def _optimize_lambda_entropy(self):
        """Select lambda via minimax over entropy strata."""
        # Define entropy boundaries on tune set
        tune_entropy = calculate_entropy(self.logits_tune, self.temperature)
        th1, th2 = np.quantile(tune_entropy, [0.33, 0.66])
        
        print(f"      Entropy boundaries: [{th1:.3f}, {th2:.3f}]")
        
        # Apply to val set
        val_entropy = calculate_entropy(self.logits_val, self.temperature)
        strata_idxs = [
            np.where(val_entropy <= th1)[0],
            np.where((val_entropy > th1) & (val_entropy <= th2))[0],
            np.where(val_entropy > th2)[0]
        ]
        
        print(f"      Val strata: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # Lambda selection via minimax
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on val set
            probs_val = torch.softmax(self.logits_val / self.temperature, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            # Coverage per stratum
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
            probs = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star,
                lamda=self.lamda_star, k_reg=self.k_reg,
                randomized=self.randomized, allow_zero_sets=False
            )
            
            return logits, prediction_sets