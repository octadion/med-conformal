"""
Conformal Prediction Wrapper (UPDATED)

Changes from previous version:
- Renamed MondrianCP → StratifiedCP (entropy-based group-conditional CP)
- Added ClassMondrianCP (true class-conditional Mondrian CP per Vovk 2005)
- All other methods unchanged

Key Features:
1. Randomized calibration scores (matching original RAPS paper)
2. allow_zero_sets parameter for validation experiments
3. Multi-alpha support for sensitivity analysis
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from typing import List, Tuple, Optional


def get_logits_labels(model, loader, device='cuda'):
    """Extract logits and labels from a dataloader."""
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            logits_list.append(out.cpu())
            # Ensure labels are always 1D (handles scalar and multi-dim cases)
            y = y.cpu().view(-1)
            labels_list.append(y)
    return torch.cat(logits_list), torch.cat(labels_list)


def calculate_entropy(logits):
    """Compute predictive entropy from logits."""
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return entropy(probs, axis=1)


# =============================================================================
# CORE SCORE COMPUTATION (WITH RANDOMIZATION)
# =============================================================================

def compute_aps_scores_randomized(logits: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   temperature: float = 1.0,
                                   randomized: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute APS conformity scores WITH randomization (matching original paper).
    
    From Romano et al. (2020):
    - If true label is rank 0 (top-1): score = U * p_y
    - If true label is rank k > 0: score = U * p_y + sum(p_j for j < k)
    
    Where U ~ Uniform(0,1) when randomized=True.
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
        
        if randomized:
            U = np.random.random()
        else:
            U = 1.0
        
        if rank == 0:
            scores[i] = U * p_y
        else:
            cumsum_before = cumsum[i, rank - 1].item()
            scores[i] = U * p_y + cumsum_before
    
    return scores, sorted_probs.numpy(), indices.numpy()


def compute_raps_scores_randomized(logits: torch.Tensor, 
                                    labels: torch.Tensor, 
                                    lamda: float,
                                    k_reg: int,
                                    temperature: float = 1.0,
                                    randomized: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RAPS conformity scores WITH randomization (matching original paper).
    
    RAPS score = APS score + cumulative penalty
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
        
        if randomized:
            U = np.random.random()
        else:
            U = 1.0
        
        if rank == 0:
            scores[i] = U * p_y + penalty
        else:
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
    
    Args:
        probs: Softmax probabilities (N, K)
        q_hat: Calibrated quantile threshold
        lamda: RAPS regularization (0 for APS)
        k_reg: Regularization starting point
        randomized: If True, use randomized set sizes
        allow_zero_sets: If True, allow empty prediction sets
                        (Setting to True gives exact 90% coverage!)
    
    Returns:
        List of prediction sets (numpy arrays of class indices)
    """
    n_samples, n_classes = probs.shape
    
    # Sort probabilities descending
    I = probs.argsort(axis=1)[:, ::-1]
    ordered = np.sort(probs, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    
    # Flat penalty after k_reg (like original)
    penalties = np.zeros(n_classes)
    penalties[k_reg:] = lamda
    penalties_cumsum = np.cumsum(penalties)
    
    # Base set sizes
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
    
    # KEY PARAMETER: allow_zero_sets
    # When False (default): minimum size = 1 (clinical safety)
    # When True: allows size = 0 (exact coverage validation)
    if not allow_zero_sets:
        sizes = np.maximum(sizes, 1)
    
    return [I[i, :sizes[i]] for i in range(n_samples)]


# =============================================================================
# CONFORMAL METHODS WITH allow_zero_sets SUPPORT
# =============================================================================

class StandardLAC(nn.Module):
    """Label-Conditional Conformal Prediction (Threshold-based)."""
    
    def __init__(self, model, calib_loader, alpha=0.1, 
                 allow_zero_sets=False, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.allow_zero_sets = allow_zero_sets
        self.device = device
        
        print(f"   [LAC] Calibrating (allow_zero_sets={allow_zero_sets})...")
        logits, labels = get_logits_labels(model, calib_loader, device)
        probs = torch.softmax(logits, dim=1)
        
        true_probs = probs[torch.arange(len(labels)), labels.long()]
        scores = 1 - true_probs.numpy()
        
        self.q_hat = compute_quantile(scores, alpha)
        print(f"   [LAC] Threshold: {1 - self.q_hat:.4f}")

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            threshold = 1 - self.q_hat
            prediction_sets = []
            for p in probs:
                pred_set = np.where(p >= threshold)[0]
                if len(pred_set) == 0 and not self.allow_zero_sets:
                    pred_set = np.array([np.argmax(p)])
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets


class StandardAPS(nn.Module):
    """
    Adaptive Prediction Sets (APS) with allow_zero_sets support.
    """
    
    def __init__(self, model, calib_loader, alpha=0.1, randomized=True, 
                 allow_zero_sets=False, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.device = device
        
        print(f"   [APS] Calibrating (randomized={randomized}, allow_zero_sets={allow_zero_sets})...")
        logits, labels = get_logits_labels(model, calib_loader, device)
        
        scores, _, _ = compute_aps_scores_randomized(
            logits, labels, temperature=1.0, randomized=randomized
        )
        
        self.q_hat = compute_quantile(scores, alpha)
        
        print(f"   [APS] Score distribution: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")
        print(f"   [APS] q_hat: {self.q_hat:.4f}")

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat, 
                lamda=0.0, k_reg=2,
                randomized=self.randomized,
                allow_zero_sets=self.allow_zero_sets
            )
            
            return logits, prediction_sets


class SizeOptimizedRAPS(nn.Module):
    """
    RAPS with lambda optimized for minimum average set size.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader, 
                 alpha=0.1, k_reg=2, randomized=True, 
                 allow_zero_sets=False, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.device = device
        self.num_classes = model.num_classes
        
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [RAPS-Std] Optimizing lambda (allow_zero_sets={allow_zero_sets})...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_size()
        print(f"   [RAPS-Std] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")

    def _get_qhat(self, logits, labels, lamda):
        scores, _, _ = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg, 
            temperature=1.0, randomized=self.randomized
        )
        return compute_quantile(scores, self.alpha)

    def _optimize_lambda_size(self):
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_size = float('inf')
        target_cov = 1 - self.alpha
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            probs_val = torch.softmax(self.logits_val, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                randomized=self.randomized, 
                allow_zero_sets=self.allow_zero_sets
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
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star, 
                lamda=self.lamda_star, k_reg=self.k_reg,
                randomized=self.randomized, 
                allow_zero_sets=self.allow_zero_sets
            )
            
            return logits, prediction_sets


class EntropyStratifiedRAPS(nn.Module):
    """
    Entropy-Stratified RAPS with allow_zero_sets support.
    
    Key contribution: Selects λ via minimax optimization over entropy strata.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader, 
                 alpha=0.1, k_reg=2, randomized=True, 
                 allow_zero_sets=False, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.device = device
        self.num_classes = model.num_classes
        
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [Entropy-RAPS] Optimizing lambda (allow_zero_sets={allow_zero_sets})...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"   [Entropy-RAPS] λ*={self.lamda_star:.5f}, q_hat={self.q_hat_star:.4f}")

    def _get_qhat(self, logits, labels, lamda):
        scores, _, _ = compute_raps_scores_randomized(
            logits, labels, lamda, self.k_reg,
            temperature=1.0, randomized=self.randomized
        )
        return compute_quantile(scores, self.alpha)

    def _optimize_lambda_entropy(self):
        # Define entropy boundaries on tune set
        tune_entropy = calculate_entropy(self.logits_tune)
        th1, th2 = np.quantile(tune_entropy, [0.33, 0.66])
        self.entropy_boundaries = (th1, th2)
        print(f"      Entropy boundaries: [{th1:.3f}, {th2:.3f}]")
        
        # Apply to val set
        val_entropy = calculate_entropy(self.logits_val)
        strata_idxs = [
            np.where(val_entropy <= th1)[0],
            np.where((val_entropy > th1) & (val_entropy <= th2))[0],
            np.where(val_entropy > th2)[0]
        ]
        print(f"      Val strata sizes: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # Lambda selection via minimax
        lambda_grid = np.linspace(0, 0.1, 30)
        best_lam, best_q = 0.0, 0.0
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            probs_val = torch.softmax(self.logits_val, dim=1).cpu().numpy()
            
            pred_sets = construct_prediction_sets_gcq(
                probs_val, q_cand, lam, self.k_reg,
                randomized=self.randomized, 
                allow_zero_sets=self.allow_zero_sets
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
            
            prediction_sets = construct_prediction_sets_gcq(
                probs, self.q_hat_star,
                lamda=self.lamda_star, k_reg=self.k_reg,
                randomized=self.randomized, 
                allow_zero_sets=self.allow_zero_sets
            )
            
            return logits, prediction_sets


class StratifiedCP(nn.Module):
    """
    Stratified (Entropy-Based Group-Conditional) Conformal Prediction.
    
    Previously named MondrianCP. Renamed for clarity because this method
    stratifies by ENTROPY TERTILES, not by class labels.
    
    Standard Mondrian CP (Vovk 2005) conditions on class labels.
    This method conditions on entropy-based difficulty strata instead.
    
    Kept for comparison as an alternative group-conditional approach.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader,
                 alpha=0.1, n_strata=3, randomized=True, 
                 allow_zero_sets=False, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.n_strata = n_strata
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.device = device
        self.num_classes = model.num_classes
        
        print(f"   [Stratified-CP] Calibrating (allow_zero_sets={allow_zero_sets})...")
        
        logits_tune, _ = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        
        # Define entropy boundaries from tune set
        tune_entropy = calculate_entropy(logits_tune)
        quantiles = np.linspace(0, 1, n_strata + 1)[1:-1]
        self.entropy_boundaries = np.quantile(tune_entropy, quantiles)
        
        # Per-stratum quantiles using calib set
        cal_entropy = calculate_entropy(self.logits_cal)
        cal_strata = self._assign_to_strata(cal_entropy)
        
        self.stratum_quantiles = {}
        for s in range(n_strata):
            mask = (cal_strata == s)
            n_s = mask.sum()
            
            if n_s == 0:
                self.stratum_quantiles[s] = 1.0
                continue
            
            stratum_logits = self.logits_cal[mask]
            stratum_labels = self.labels_cal[mask]
            
            scores, _, _ = compute_aps_scores_randomized(
                stratum_logits, stratum_labels, 
                temperature=1.0, randomized=randomized
            )
            
            self.stratum_quantiles[s] = compute_quantile(scores, alpha)
            print(f"      Stratum {s}: n={n_s}, q_hat={self.stratum_quantiles[s]:.4f}")

    def _assign_to_strata(self, entropy_values):
        strata = np.zeros(len(entropy_values), dtype=int)
        for i, ent in enumerate(entropy_values):
            stratum = 0
            for boundary in self.entropy_boundaries:
                if ent > boundary:
                    stratum += 1
            strata[i] = min(stratum, self.n_strata - 1)
        return strata

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            test_entropy = calculate_entropy(logits)
            test_strata = self._assign_to_strata(test_entropy)
            
            I = probs.argsort(axis=1)[:, ::-1]
            ordered = np.sort(probs, axis=1)[:, ::-1]
            cumsum = np.cumsum(ordered, axis=1)
            
            prediction_sets = []
            for i in range(len(probs)):
                q_hat = self.stratum_quantiles[test_strata[i]]
                
                sizes_base = (cumsum[i] <= q_hat).sum() + 1
                sizes_base = min(sizes_base, probs.shape[1])
                
                if self.randomized and sizes_base > 1:
                    idx = sizes_base - 1
                    if ordered[i, idx] > 1e-10:
                        score_without_last = cumsum[i, idx] - ordered[i, idx]
                        V = (q_hat - score_without_last) / ordered[i, idx]
                        V = np.clip(V, 0.0, 1.0)
                        if np.random.random() >= V:
                            sizes_base -= 1
                
                # Apply allow_zero_sets
                if not self.allow_zero_sets:
                    sizes_base = max(sizes_base, 1)
                
                prediction_sets.append(I[i, :sizes_base])
            
            return logits, prediction_sets


# Backward compatibility alias
MondrianCP = StratifiedCP


# =============================================================================
# NEW: ClassMondrianCP (True Class-Conditional Mondrian CP)
# =============================================================================

class ClassMondrianCP(nn.Module):
    """
    True Class-Conditional Mondrian Conformal Prediction (Vovk 2005).
    
    Groups calibration samples by PREDICTED CLASS (ŷ = argmax softmax),
    then computes a separate APS quantile per class. At test time, the
    prediction set for sample x is constructed using q_hat_{ŷ(x)}.
    
    This is the standard Mondrian CP baseline that reviewers expect.
    
    Key differences from StratifiedCP (entropy-based):
    - Groups = K classes (e.g., 9 for PathMNIST, 11 for OrganAMNIST)
    - No entropy boundaries needed
    - No tune_loader required (only calib_loader)
    - No λ parameter (pure APS scoring)
    - Per-class calibration sizes vary by class frequency
    
    Fallback: If a class has fewer than min_class_size calibration samples,
    the global (pooled) quantile is used instead.
    
    References:
        Vovk, V. (2005). "Algorithmic Learning in a Random World"
        Romano, Y., Sesia, M., & Candès, E. J. (2020). 
            "Classification with Valid and Adaptive Coverage" (NeurIPS)
    """
    
    def __init__(self, model, calib_loader,
                 alpha=0.1,
                 randomized=True,
                 allow_zero_sets=False,
                 min_class_size=5,
                 device='cuda',
                 # Accept but ignore these for API compatibility with runner
                 tune_loader=None,
                 val_loader=None):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.min_class_size = min_class_size
        self.device = device
        self.num_classes = model.num_classes
        
        print(f"   [Class-Mondrian] Calibrating (allow_zero_sets={allow_zero_sets})...")
        print(f"   [Class-Mondrian] num_classes={self.num_classes}, "
              f"min_class_size={min_class_size}")
        
        # Extract logits and labels from calibration set
        logits_cal, labels_cal = get_logits_labels(model, calib_loader, device)
        
        # Compute predicted classes on calibration set
        probs_cal = torch.softmax(logits_cal, dim=1)
        predicted_classes = torch.argmax(probs_cal, dim=1).numpy()
        
        # Compute ALL calibration APS scores (for global fallback)
        all_scores, _, _ = compute_aps_scores_randomized(
            logits_cal, labels_cal,
            temperature=1.0, randomized=randomized
        )
        self.global_q_hat = compute_quantile(all_scores, alpha)
        
        # Per-class quantile calibration
        self.class_quantiles = {}
        self.class_sizes = {}
        fallback_classes = []
        
        for k in range(self.num_classes):
            # Select samples where PREDICTED class is k
            mask = (predicted_classes == k)
            n_k = mask.sum()
            self.class_sizes[k] = int(n_k)
            
            if n_k < min_class_size:
                # Fallback to global quantile
                self.class_quantiles[k] = self.global_q_hat
                fallback_classes.append(k)
                print(f"      Class {k}: n={n_k} < {min_class_size} → "
                      f"FALLBACK to global q_hat={self.global_q_hat:.4f}")
            else:
                # Compute per-class APS scores
                class_logits = logits_cal[mask]
                class_labels = labels_cal[mask]
                
                class_scores, _, _ = compute_aps_scores_randomized(
                    class_logits, class_labels,
                    temperature=1.0, randomized=randomized
                )
                
                self.class_quantiles[k] = compute_quantile(class_scores, alpha)
                print(f"      Class {k}: n={n_k}, q_hat={self.class_quantiles[k]:.4f}")
        
        # Summary
        n_fallback = len(fallback_classes)
        print(f"   [Class-Mondrian] Global fallback q_hat: {self.global_q_hat:.4f}")
        if n_fallback > 0:
            print(f"   [Class-Mondrian] ⚠ {n_fallback}/{self.num_classes} classes "
                  f"using fallback: {fallback_classes}")
        else:
            print(f"   [Class-Mondrian] ✓ All {self.num_classes} classes have "
                  f"sufficient calibration samples")
        
        # Store calibration info for later analysis
        self.calibration_info = {
            'class_sizes': self.class_sizes,
            'class_quantiles': {k: float(v) for k, v in self.class_quantiles.items()},
            'global_q_hat': float(self.global_q_hat),
            'fallback_classes': fallback_classes,
            'total_calib_samples': len(labels_cal)
        }
    
    def forward(self, x):
        """
        Construct prediction sets using class-conditional quantiles.
        
        For each test sample:
        1. Compute softmax probabilities
        2. Determine predicted class ŷ = argmax(probs)
        3. Use q_hat_ŷ as the threshold
        4. Build prediction set via APS-style cumulative sum
        """
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Predicted classes for test batch
            predicted_classes = np.argmax(probs, axis=1)
            
            # Sort probabilities descending for set construction
            I = probs.argsort(axis=1)[:, ::-1]
            ordered = np.sort(probs, axis=1)[:, ::-1]
            cumsum = np.cumsum(ordered, axis=1)
            
            prediction_sets = []
            for i in range(len(probs)):
                # Get class-specific quantile
                pred_class = predicted_classes[i]
                q_hat = self.class_quantiles.get(pred_class, self.global_q_hat)
                
                # APS-style set construction (no RAPS penalty)
                sizes_base = (cumsum[i] <= q_hat).sum() + 1
                sizes_base = min(sizes_base, probs.shape[1])
                
                # Randomized set size
                if self.randomized and sizes_base > 1:
                    idx = sizes_base - 1
                    if ordered[i, idx] > 1e-10:
                        score_without_last = cumsum[i, idx] - ordered[i, idx]
                        V = (q_hat - score_without_last) / ordered[i, idx]
                        V = np.clip(V, 0.0, 1.0)
                        if np.random.random() >= V:
                            sizes_base -= 1
                
                # Apply allow_zero_sets
                if not self.allow_zero_sets:
                    sizes_base = max(sizes_base, 1)
                
                prediction_sets.append(I[i, :sizes_base])
            
            return logits, prediction_sets
    
    def get_calibration_summary(self) -> str:
        """Return a formatted summary of per-class calibration for paper."""
        lines = []
        lines.append(f"Class-Conditional Mondrian CP Calibration Summary")
        lines.append(f"{'Class':<8} {'N_calib':<10} {'q_hat':<10} {'Fallback':<10}")
        lines.append("-" * 40)
        for k in range(self.num_classes):
            n_k = self.class_sizes[k]
            q_k = self.class_quantiles[k]
            fb = "Yes" if k in self.calibration_info['fallback_classes'] else "No"
            lines.append(f"{k:<8} {n_k:<10} {q_k:<10.4f} {fb:<10}")
        lines.append("-" * 40)
        lines.append(f"Global q_hat: {self.global_q_hat:.4f}")
        lines.append(f"Total calib samples: {self.calibration_info['total_calib_samples']}")
        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def predict_with_sets(conformal_model, dataloader):
    """Helper function for evaluation."""
    device = getattr(conformal_model, 'device', 'cuda')
    conformal_model.eval()
    
    all_logits, all_preds, all_probs, all_sets, all_labels = [], [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits, sets = conformal_model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            if len(y.shape) > 1:
                y = y.squeeze()
            all_labels.append(y.cpu().numpy())
            all_sets.extend(sets)

    return (
        np.concatenate(all_logits),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        all_sets,
        np.concatenate(all_labels)
    )