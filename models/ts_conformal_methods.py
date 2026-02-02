"""
Temperature-Scaled Conformal Prediction Methods

This module provides temperature-scaled variants of conformal prediction methods.
Temperature scaling improves probability calibration before computing prediction sets.

Baseline comparison requested by Stanford reviewer (Point #3).
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
from models.randomization_utils import compute_randomized_quantile


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


# ============================================================================
# TEMPERATURE-SCALED APS
# ============================================================================

class TemperatureScaledAPS(nn.Module):
    """
    APS with temperature scaling.
    
    Protocol:
    1. Calibrate temperature T on calibration set
    2. Apply APS with temperature-scaled probabilities
    
    This baseline requested by Stanford reviewer to compare with
    entropy stratification approach.
    """
    
    def __init__(self, model, calib_loader, alpha=0.1, 
                 ts_max_iters=50, device='cuda'):
        """
        Args:
            model: Trained classifier
            calib_loader: Calibration data
            alpha: Miscoverage level
            ts_max_iters: Temperature scaling iterations
            device: cuda or cpu
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.device = device
        
        print("   [TS-APS] Step 1: Calibrating temperature...")
        
        # Calibrate temperature
        self.temp_scaler = TemperatureScaler(
            model, calib_loader, device=device,
            max_iters=ts_max_iters, verbose=False
        )
        self.temperature = self.temp_scaler.temperature
        
        print(f"   [TS-APS] Optimal T = {self.temperature:.4f}")
        print("   [TS-APS] Step 2: Computing APS quantile with scaled probabilities...")
        
        # Compute APS quantile with temperature-scaled logits
        logits, labels = get_logits_labels(model, calib_loader, device)
        scores = self._compute_aps_scores(logits, labels, self.temperature)
        
        n = len(labels)
        self.q_hat = np.quantile(
            scores, 
            np.ceil((n + 1) * (1 - alpha)) / n, 
            method='higher'
        )
        
        print(f"   [TS-APS] Q_hat: {self.q_hat:.4f}")
    
    def _compute_aps_scores(self, logits, labels, temperature):
        """Compute APS scores with temperature-scaled probabilities."""
        probs = torch.softmax(logits / temperature, dim=1)
        sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        
        rows = torch.arange(len(labels))
        ranks = torch.zeros(len(labels), dtype=torch.long)
        for i, lbl in enumerate(labels):
            rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
            ranks[i] = rank.item() if len(rank) > 0 else 0
        
        scores = cumsum[rows, ranks]
        return scores.numpy()
    
    def forward(self, x):
        """Forward pass with temperature-scaled probabilities."""
        with torch.no_grad():
            logits = self.model(x)
            
            # Apply temperature scaling
            probs = torch.softmax(logits / self.temperature, dim=1)
            sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            for i in range(x.size(0)):
                mask = cumsum[i] <= self.q_hat
                if mask.sum() == 0:
                    k_star = 0
                else:
                    k_star = mask.nonzero(as_tuple=True)[0][-1].item()
                
                pred_set = idx[i, :k_star+1].cpu().numpy()
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets


# ============================================================================
# TEMPERATURE-SCALED RAPS (Size Optimized)
# ============================================================================

class TemperatureScaledRAPS(nn.Module):
    """
    Size-optimized RAPS with temperature scaling.
    
    Protocol:
    1. Calibrate temperature T
    2. Select λ to minimize average set size (with T-scaled probabilities)
    3. Apply RAPS with selected λ and T
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, ts_max_iters=50, device='cuda'):
        """
        Args:
            model: Trained classifier
            tune_loader: Unused (kept for API consistency)
            calib_loader: For temperature + quantile calibration
            val_loader: For lambda selection
            alpha: Miscoverage level
            k_reg: RAPS regularization starting point
            ts_max_iters: Temperature scaling iterations
            device: cuda or cpu
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.device = device
        self.num_classes = model.num_classes
        
        print("   [TS-RAPS] Step 1: Calibrating temperature...")
        
        # Calibrate temperature on calibration set
        self.temp_scaler = TemperatureScaler(
            model, calib_loader, device=device,
            max_iters=ts_max_iters, verbose=False
        )
        self.temperature = self.temp_scaler.temperature
        
        print(f"   [TS-RAPS] Optimal T = {self.temperature:.4f}")
        
        # Extract logits
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print("   [TS-RAPS] Step 2: Optimizing lambda for minimal set size...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_size()
        print(f"   [TS-RAPS] Lambda: {self.lamda_star:.5f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute RAPS quantile with temperature-scaled probabilities."""
        probs = torch.softmax(logits / self.temperature, dim=1)
        sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        
        rows = torch.arange(len(labels))
        ranks = torch.zeros(len(labels), dtype=torch.long)
        for i, lbl in enumerate(labels):
            rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
            ranks[i] = rank.item() if len(rank) > 0 else 0
        
        scores = cumsum[rows, ranks] + lamda * torch.clamp(ranks - self.k_reg + 1, min=0)
        n = len(labels)
        return np.quantile(
            scores.numpy(), 
            np.ceil((n + 1) * (1 - self.alpha)) / n, 
            method='higher'
        )
    
    def _optimize_lambda_size(self):
        """Select lambda to minimize average set size."""
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q = 0.0, 0.0
        min_size = float('inf')
        target_cov = 1 - self.alpha
        
        for lam in lambda_grid:
            # Compute quantile on calibration set
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on validation set with temperature scaling
            probs_val = torch.softmax(self.logits_val / self.temperature, dim=1)
            sorted_probs, idx = torch.sort(probs_val, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            # Compute sizes
            penalties = lam * torch.clamp(
                torch.arange(self.num_classes) - self.k_reg + 1, min=0
            )
            scores_all = cumsum + penalties.unsqueeze(0)
            sizes = (scores_all <= q_cand).sum(dim=1)
            sizes = torch.clamp(sizes, min=1)
            avg_size = sizes.float().mean().item()
            
            # Check coverage
            rows = torch.arange(len(self.labels_val))
            ranks = torch.zeros(len(self.labels_val), dtype=torch.long)
            for i, lbl in enumerate(self.labels_val):
                rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
                ranks[i] = rank.item() if len(rank) > 0 else 0
            
            scores_true = cumsum[rows, ranks] + lam * torch.clamp(
                ranks - self.k_reg + 1, min=0
            )
            coverage = (scores_true <= q_cand).float().mean().item()
            
            # Select lambda with smallest size (if coverage adequate)
            if coverage >= target_cov - 0.005:
                if avg_size < min_size:
                    min_size = avg_size
                    best_lam = lam
                    best_q = q_cand
        
        return best_lam, best_q
    
    def forward(self, x):
        """Forward pass with temperature-scaled RAPS."""
        with torch.no_grad():
            logits = self.model(x)
            
            # Apply temperature scaling
            probs = torch.softmax(logits / self.temperature, dim=1)
            sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            for i in range(x.size(0)):
                penalties = self.lamda_star * torch.clamp(
                    torch.arange(self.num_classes, device=x.device) - self.k_reg + 1,
                    min=0
                )
                scores = cumsum[i] + penalties
                
                mask = scores <= self.q_hat_star
                if mask.sum() == 0:
                    k_star = 0
                else:
                    k_star = mask.nonzero()[-1].item()
                
                pred_set = idx[i, :k_star+1].cpu().numpy()
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets


# ============================================================================
# TEMPERATURE-SCALED ENTROPY-RAPS (Ours + TS)
# ============================================================================

class TemperatureScaledEntropyRAPS(nn.Module):
    """
    Entropy-stratified RAPS with temperature scaling.
    
    This combines two orthogonal improvements:
    1. Temperature scaling for better calibration
    2. Entropy stratification for worst-case protection
    
    Protocol:
    1. Calibrate temperature T on calibration set
    2. Define entropy boundaries on tune set (with T-scaled probabilities)
    3. Select λ via minimax over strata (with T-scaled probabilities)
    4. Apply RAPS with selected λ and T
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, k_reg=2, ts_max_iters=50, device='cuda'):
        """
        Args:
            model: Trained classifier
            tune_loader: Define entropy boundaries
            calib_loader: Temperature + quantile calibration
            val_loader: Lambda selection via minimax
            alpha: Miscoverage level
            k_reg: RAPS regularization
            ts_max_iters: Temperature scaling iterations
            device: cuda or cpu
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.device = device
        self.num_classes = model.num_classes
        
        print("   [TS-EntropyRAPS] Step 1: Calibrating temperature...")
        
        # Calibrate temperature
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
        
        print("   [TS-EntropyRAPS] Step 2: Optimizing lambda for stratified safety...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"   [TS-EntropyRAPS] Lambda: {self.lamda_star:.5f}")
    
    def _get_qhat(self, logits, labels, lamda):
        """Compute RAPS quantile with temperature-scaled probabilities."""
        probs = torch.softmax(logits / self.temperature, dim=1)
        sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        
        rows = torch.arange(len(labels))
        ranks = torch.zeros(len(labels), dtype=torch.long)
        for i, lbl in enumerate(labels):
            rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
            ranks[i] = rank.item() if len(rank) > 0 else 0
        
        scores = cumsum[rows, ranks] + lamda * torch.clamp(ranks - self.k_reg + 1, min=0)
        n = len(labels)
        return np.quantile(
            scores.numpy(),
            np.ceil((n + 1) * (1 - self.alpha)) / n,
            method='higher'
        )
    
    def _optimize_lambda_entropy(self):
        """Select lambda via minimax over entropy strata."""
        # Define entropy boundaries on tune set (with temperature scaling)
        tune_entropy = calculate_entropy(self.logits_tune, self.temperature)
        th1, th2 = np.quantile(tune_entropy, [0.33, 0.66])
        
        print(f"      Entropy boundaries (T-scaled): [{th1:.3f}, {th2:.3f}]")
        
        # Apply to validation set
        val_entropy = calculate_entropy(self.logits_val, self.temperature)
        strata_idxs = [
            np.where(val_entropy <= th1)[0],
            np.where((val_entropy > th1) & (val_entropy <= th2))[0],
            np.where(val_entropy > th2)[0]
        ]
        
        print(f"      Val set strata: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # Lambda selection via minimax
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q = 0.0, 0.0
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            # Compute quantile on calibration set
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # Evaluate on validation set with temperature scaling
            probs_val = torch.softmax(self.logits_val / self.temperature, dim=1)
            sorted_probs, idx = torch.sort(probs_val, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            rows = torch.arange(len(self.labels_val))
            ranks = torch.zeros(len(self.labels_val), dtype=torch.long)
            for i, lbl in enumerate(self.labels_val):
                rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
                ranks[i] = rank.item() if len(rank) > 0 else 0
            
            scores_val = cumsum[rows, ranks] + lam * torch.clamp(
                ranks - self.k_reg + 1, min=0
            )
            is_covered = (scores_val <= q_cand).numpy().astype(int)
            
            # Compute average size for tie-breaking
            penalties = lam * torch.clamp(
                torch.arange(self.num_classes) - self.k_reg + 1, min=0
            )
            scores_all = cumsum + penalties.unsqueeze(0)
            sizes = (scores_all <= q_cand).sum(dim=1)
            curr_avg_size = sizes.float().mean().item()
            
            # Compute worst-case violation
            violations = []
            for s_idx in strata_idxs:
                if len(s_idx) == 0: continue
                cov = np.mean(is_covered[s_idx])
                violations.append(abs(cov - (1 - self.alpha)))
            max_viol = max(violations) if violations else 1.0
            
            # Minimax selection
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
        """Forward pass with temperature-scaled entropy-RAPS."""
        with torch.no_grad():
            logits = self.model(x)
            
            # Apply temperature scaling
            probs = torch.softmax(logits / self.temperature, dim=1)
            sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            for i in range(x.size(0)):
                penalties = self.lamda_star * torch.clamp(
                    torch.arange(self.num_classes, device=x.device) - self.k_reg + 1,
                    min=0
                )
                scores = cumsum[i] + penalties
                
                mask = scores <= self.q_hat_star
                if mask.sum() == 0:
                    k_star = 0
                else:
                    k_star = mask.nonzero()[-1].item()
                
                pred_set = idx[i, :k_star+1].cpu().numpy()
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets
