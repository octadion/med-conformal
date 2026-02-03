import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from typing import List, Tuple, Dict

# Import randomization utilities for exact coverage
import sys
sys.path.append('.')
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
            # Handle label dimension
            if len(y.shape) > 1:
                y = y.squeeze()
            labels_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(labels_list)

def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return entropy(probs, axis=1)

def compute_aps_scores_batch(logits, labels, temperature=1.0):
    """
    Compute APS conformity scores for a batch.
    
    Args:
        logits: Model outputs (N, K)
        labels: True labels (N,)
        temperature: Temperature scaling (default 1.0 = no scaling)
    
    Returns:
        scores: Conformity scores (N,)
        sorted_probs: Sorted probabilities (N, K)
        indices: Sorting indices (N, K)
    """
    probs = torch.softmax(logits / temperature, dim=1)
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    # Find rank of true label
    rows = torch.arange(len(labels))
    ranks = torch.zeros(len(labels), dtype=torch.long)
    for i, lbl in enumerate(labels):
        rank = (indices[i] == lbl).nonzero(as_tuple=True)[0]
        ranks[i] = rank.item() if len(rank) > 0 else 0
    
    scores = cumsum[rows, ranks]
    return scores.numpy(), sorted_probs.numpy(), indices.numpy()


def compute_raps_scores_batch(logits, labels, lamda, k_reg, temperature=1.0):
    """
    Compute RAPS conformity scores for a batch.
    
    Args:
        logits: Model outputs (N, K)
        labels: True labels (N,)
        lamda: RAPS regularization parameter
        k_reg: Regularization starting point
        temperature: Temperature scaling
    
    Returns:
        scores: Conformity scores (N,)
        sorted_probs: Sorted probabilities (N, K)
        indices: Sorting indices (N, K)
    """
    probs = torch.softmax(logits / temperature, dim=1)
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    # Find rank of true label
    rows = torch.arange(len(labels))
    ranks = torch.zeros(len(labels), dtype=torch.long)
    for i, lbl in enumerate(labels):
        rank = (indices[i] == lbl).nonzero(as_tuple=True)[0]
        ranks[i] = rank.item() if len(rank) > 0 else 0
    
    # Add RAPS penalty
    penalty = lamda * torch.clamp(ranks - k_reg + 1, min=0)
    scores = cumsum[rows, ranks] + penalty
    
    return scores.numpy(), sorted_probs.numpy(), indices.numpy()


def construct_prediction_sets_randomized(sorted_probs, indices, q_hat, 
                                         lamda=0.0, k_reg=2, randomized=True):
    """
    ✨ NEW: Construct prediction sets with optional randomization.
    
    This implements the randomized quantile method from Romano et al. (2020)
    to achieve exact finite-sample coverage instead of conservative coverage.
    
    Args:
        sorted_probs: Sorted probabilities (N, K) - descending order
        indices: Class indices corresponding to sorted probs (N, K)
        q_hat: Calibrated quantile threshold
        lamda: RAPS regularization (0 for APS)
        k_reg: RAPS regularization starting point
        randomized: If True, use randomized inclusion (exact coverage)
    
    Returns:
        prediction_sets: List of numpy arrays (one per sample)
    """
    n_samples, n_classes = sorted_probs.shape
    cumsum = np.cumsum(sorted_probs, axis=1)
    
    # Compute penalties (0 for APS, nonzero for RAPS)
    if lamda > 0:
        penalties = lamda * np.maximum(np.arange(n_classes) - k_reg + 1, 0)
        penalties_cumsum = np.cumsum(penalties)
    else:
        penalties_cumsum = np.zeros(n_classes)
    
    # Base set sizes (before randomization)
    scores_all = cumsum + penalties_cumsum[None, :]
    sizes_base = (scores_all <= q_hat).sum(axis=1)
    sizes_base = np.maximum(sizes_base, 1)  # At least size 1
    
    prediction_sets = []
    
    for i in range(n_samples):
        size_base = sizes_base[i]
        
        if randomized and size_base > 1:
            # Compute randomization probability V
            # V = (q_hat - score_without_last) / prob_last
            last_idx = size_base - 1
            score_without_last = cumsum[i, last_idx - 1] if last_idx > 0 else 0.0
            score_without_last += penalties_cumsum[last_idx]
            
            prob_last = sorted_probs[i, last_idx]
            
            if prob_last > 1e-10:  # Avoid division by zero
                V = (q_hat - score_without_last) / prob_last
                V = np.clip(V, 0.0, 1.0)  # Ensure valid probability
                
                # Randomized inclusion
                u = np.random.uniform(0, 1)
                if u >= V:
                    # Exclude last element
                    final_size = size_base - 1
                else:
                    # Include last element
                    final_size = size_base
            else:
                final_size = size_base
        else:
            final_size = size_base
        
        # Ensure at least size 1
        final_size = max(final_size, 1)
        
        # Extract prediction set
        pred_set = indices[i, :final_size]
        prediction_sets.append(pred_set)
    
    return prediction_sets

class StandardLAC(nn.Module):
    def __init__(self, model, calib_loader, alpha=0.1, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.device = device
        
        print("   [LAC] Calibrating...")
        logits, labels = get_logits_labels(model, calib_loader, device)
        probs = torch.softmax(logits, dim=1)
        
        true_probs = probs[torch.arange(len(labels)), labels.long()]
        scores = 1 - true_probs
        
        n = len(labels)
        q_val = np.quantile(scores.numpy(), np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
        
        self.q_hat = q_val
        print(f"   [LAC] Threshold (1 - q_hat): {1 - self.q_hat:.4f}")

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            
            threshold = 1 - self.q_hat
            
            prediction_sets = []
            for p in probs.cpu().numpy():
                pred_set = np.where(p >= threshold)[0]
                if len(pred_set) == 0: 
                     pred_set = np.array([np.argmax(p)])
                prediction_sets.append(pred_set)
                
            return logits, prediction_sets


class StandardAPS(nn.Module):
    def __init__(self, model, calib_loader, alpha=0.1, randomized=True, device='cuda'):
        """
        Standard Adaptive Prediction Sets with optional randomization.
        
        Args:
            model: Trained classifier
            calib_loader: Calibration data
            alpha: Miscoverage level
            randomized: If True, use randomized quantile for exact coverage
            device: cuda or cpu
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.device = device
        
        print(f"   [APS] Calibrating (randomized={randomized})...")
        logits, labels = get_logits_labels(model, calib_loader, device)
        
        scores, _, _ = compute_aps_scores_batch(logits, labels, temperature=1.0)
        
        # Use randomized quantile for exact coverage
        self.q_hat, self.U = compute_randomized_quantile(scores, alpha, randomized)
        print(f"   [APS] Q_hat: {self.q_hat:.4f}" + 
              (f", U: {self.U:.3f}" if self.U is not None else ""))

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            # Build prediction sets with randomization
            prediction_sets = []
            for i in range(x.size(0)):
                if not self.randomized or self.U is None:
                    # Standard (non-randomized)
                    mask = cumsum[i] <= self.q_hat
                    if mask.sum() == 0:
                        k_star = 0
                    else:
                        k_star = mask.nonzero()[-1].item()
                else:
                    # Randomized
                    below = cumsum[i] < self.q_hat
                    if below.sum() == 0:
                        k_base = 0
                    else:
                        k_base = below.nonzero()[-1].item() + 1
                    
                    # Check boundary
                    if k_base < len(cumsum[i]):
                        at_boundary = torch.isclose(
                            cumsum[i, k_base],
                            torch.tensor(self.q_hat),
                            atol=1e-6
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
# FIXED RAPS CLASSES WITH 3-WAY SPLIT
# ============================================================================

class SizeOptimizedRAPS(nn.Module):
    def __init__(self, model, tune_loader, calib_loader, val_loader, alpha=0.1, k_reg=2, randomized=True, device='cuda'):
        """
        ✨ FIXED: Proper 3-way split protocol for size-optimized RAPS.
        
        Protocol:
        1. Tune set: NOT USED (size optimization doesn't need it)
        2. Calib set: Compute q_hat(λ) for each λ candidate
        3. Val set: Evaluate average size for each λ, pick smallest
        
        Args:
            model: Trained classifier
            tune_loader: Unused (kept for API consistency)
            calib_loader: For computing quantiles
            val_loader: For selecting best λ
            alpha: Miscoverage level
            k_reg: RAPS regularization starting point
            randomized: If True, use randomized quantile for exact coverage
            device: cuda or cpu
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.device = device
        self.num_classes = model.num_classes
        
        # Extract logits - calib for quantiles, val for selection
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [RAPS-Standard] Optimizing Lambda (randomized={randomized})...")
        self.lamda_star, self.q_hat_star, self.U = self._optimize_lambda_size()
        print(f"   [RAPS-Standard] Lambda: {self.lamda_star:.5f}")

    def _get_qhat(self, logits, labels, lamda):
        """
        Compute quantile on calibration set for given lambda.
        Returns (q_hat, U) where U is randomization parameter.
        """
        scores, _, _ = compute_raps_scores_batch(
            logits, labels, lamda, self.k_reg, temperature=1.0
        )
        # Use randomized quantile
        q_hat, U = compute_randomized_quantile(scores, self.alpha, self.randomized)
        return q_hat, U

    def _optimize_lambda_size(self):
        """
        ✨ FIXED: Proper separation.
        
        For each λ:
        1. Compute q_hat(λ) on CALIB set
        2. Evaluate avg size on VAL set
        3. Pick λ with smallest size (subject to coverage constraint)
        """
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q, best_U = 0.0, 0.0, None
        min_size = float('inf')
        target_cov = 1 - self.alpha
        
        for lam in lambda_grid:
            # STEP 1: Compute quantile on CALIBRATION set
            q_cand, U_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # STEP 2: Evaluate on VALIDATION set
            probs_val = torch.softmax(self.logits_val, dim=1)
            sorted_probs, idx = torch.sort(probs_val, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            # Compute sizes on val set
            penalties = lam * torch.clamp(torch.arange(self.num_classes) - self.k_reg + 1, min=0)
            scores_all = cumsum + penalties.unsqueeze(0)
            sizes = (scores_all <= q_cand).sum(dim=1)
            sizes = torch.clamp(sizes, min=1)
            avg_size = sizes.float().mean().item()
            
            # Check coverage on val set
            rows = torch.arange(len(self.labels_val))
            ranks = torch.zeros(len(self.labels_val), dtype=torch.long)
            for i, lbl in enumerate(self.labels_val):
                 rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
                 ranks[i] = rank.item() if len(rank) > 0 else 0
            
            scores_true = cumsum[rows, ranks] + lam * torch.clamp(ranks - self.k_reg + 1, min=0)
            coverage = (scores_true <= q_cand).float().mean().item()
            
            # Select lambda with smallest size (if coverage is adequate)
            if coverage >= target_cov - 0.005: 
                if avg_size < min_size:
                    min_size = avg_size
                    best_lam = lam
                    best_q = q_cand
                    best_U = U_cand
        
        return best_lam, best_q, best_U

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            # Build prediction sets with randomization
            prediction_sets = []
            penalties = self.lamda_star * torch.clamp(
                torch.arange(self.num_classes) - self.k_reg + 1, min=0
            )
            
            for i in range(x.size(0)):
                scores = cumsum[i] + penalties
                
                if not self.randomized or self.U is None:
                    # Standard (non-randomized)
                    mask = scores <= self.q_hat_star
                    if mask.sum() == 0:
                        k_star = 0
                    else:
                        k_star = mask.nonzero()[-1].item()
                else:
                    # Randomized
                    below = scores < self.q_hat_star
                    if below.sum() == 0:
                        k_base = 0
                    else:
                        k_base = below.nonzero()[-1].item() + 1
                    
                    # Check boundary
                    if k_base < len(scores):
                        at_boundary = torch.isclose(
                            scores[k_base],
                            torch.tensor(self.q_hat_star),
                            atol=1e-6
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


class EntropyStratifiedRAPS(nn.Module):
    def __init__(self, model, tune_loader, calib_loader, val_loader, alpha=0.1, k_reg=2, randomized=True, device='cuda'):
        """
        ✨ FIXED: Proper 3-way split protocol for entropy-stratified RAPS.
        
        Protocol:
        1. Tune set: Define entropy tertile boundaries (computed ONCE)
        2. Calib set: Compute q_hat(λ) for each λ candidate
        3. Val set: Evaluate stratified coverage for each λ, pick via minimax
        
        This is the CORE CONTRIBUTION of the paper with proper validity.
        
        Args:
            model: Trained classifier
            tune_loader: For defining entropy boundaries
            calib_loader: For computing quantiles
            val_loader: For selecting best λ via minimax
            alpha: Miscoverage level
            k_reg: RAPS regularization starting point
            randomized: If True, use randomized quantile for exact coverage
            device: cuda or cpu
        """
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.randomized = randomized
        self.num_classes = model.num_classes
        
        # Extract logits from all three splits
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_val, self.labels_val = get_logits_labels(model, val_loader, device)
        
        print(f"   [Entropy-RAPS] Optimizing Lambda (randomized={randomized})...")
        self.lamda_star, self.q_hat_star, self.U = self._optimize_lambda_entropy()
        print(f"   [Entropy-RAPS] Lambda: {self.lamda_star:.5f}")

    def _get_qhat(self, logits, labels, lamda):
        """
        Compute quantile on calibration set for given lambda.
        Returns (q_hat, U) where U is randomization parameter.
        """
        scores, _, _ = compute_raps_scores_batch(
            logits, labels, lamda, self.k_reg, temperature=1.0
        )
        # Use randomized quantile
        q_hat, U = compute_randomized_quantile(scores, self.alpha, self.randomized)
        return q_hat, U

    def _optimize_lambda_entropy(self):
        """
        ✨ FIXED: Proper 3-way protocol with entropy stratification.
        
        STEP 1: Define strata boundaries on TUNE set (computed ONCE)
        STEP 2: For each λ, compute q_hat(λ) on CALIB set
        STEP 3: Evaluate stratified coverage on VAL set
        STEP 4: Select λ* via minimax over strata violations
        """
        
        # STEP 1: Define entropy boundaries on TUNE set (before λ loop!)
        tune_entropy = calculate_entropy(self.logits_tune)
        th1, th2 = np.quantile(tune_entropy, [0.33, 0.66])
        
        print(f"      Entropy boundaries (from tune set): [{th1:.3f}, {th2:.3f}]")
        
        # Apply same boundaries to VAL set
        val_entropy = calculate_entropy(self.logits_val)
        strata_idxs = [
            np.where(val_entropy <= th1)[0],
            np.where((val_entropy > th1) & (val_entropy <= th2))[0],
            np.where(val_entropy > th2)[0]
        ]
        
        print(f"      Val set strata sizes: Easy={len(strata_idxs[0])}, "
              f"Med={len(strata_idxs[1])}, Hard={len(strata_idxs[2])}")
        
        # STEP 2-4: Lambda selection loop
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q, best_U = 0.0, 0.0, None
        min_max_violation = float('inf')
        best_avg_size = float('inf')
        
        for lam in lambda_grid:
            # STEP 2: Compute quantile on CALIBRATION set
            q_cand, U_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            # STEP 3: Evaluate coverage on VALIDATION set
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
            
            # STEP 4: Compute worst-case violation across strata
            violations = []
            for s_idx in strata_idxs:
                if len(s_idx) == 0: continue
                cov = np.mean(is_covered[s_idx])
                violations.append(abs(cov - (1 - self.alpha)))
            max_viol = max(violations) if violations else 1.0
            
            # Minimax selection: minimize worst-case violation
            if max_viol < min_max_violation:
                min_max_violation = max_viol
                best_lam = lam
                best_q = q_cand
                best_U = U_cand
                best_avg_size = curr_avg_size
            elif abs(max_viol - min_max_violation) < 0.001:
                # Tie-break by size
                if curr_avg_size < best_avg_size:
                    min_max_violation = max_viol
                    best_lam = lam
                    best_q = q_cand
                    best_U = U_cand
                    best_avg_size = curr_avg_size
                
        return best_lam, best_q, best_U

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            # Build prediction sets with randomization
            prediction_sets = []
            penalties = self.lamda_star * torch.clamp(
                torch.arange(self.num_classes) - self.k_reg + 1, min=0
            )
            
            for i in range(x.size(0)):
                scores = cumsum[i] + penalties
                
                if not self.randomized or self.U is None:
                    # Standard (non-randomized)
                    mask = scores <= self.q_hat_star
                    if mask.sum() == 0:
                        k_star = 0
                    else:
                        k_star = mask.nonzero()[-1].item()
                else:
                    # Randomized
                    below = scores < self.q_hat_star
                    if below.sum() == 0:
                        k_base = 0
                    else:
                        k_base = below.nonzero()[-1].item() + 1
                    
                    # Check boundary
                    if k_base < len(scores):
                        at_boundary = torch.isclose(
                            scores[k_base],
                            torch.tensor(self.q_hat_star),
                            atol=1e-6
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

class MondrianCP(nn.Module):
    """
    Mondrian (Group-Conditional) Conformal Prediction.
    
    Uses entropy-defined strata with SEPARATE quantile calibration per group.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader, 
                 alpha=0.1, n_strata=3, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.n_strata = n_strata
        self.device = device
        self.num_classes = model.num_classes
        
        print(f"   [Mondrian-CP] Calibrating with {n_strata} entropy strata...")
        
        # Get logits
        logits_tune, _ = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        
        # Define entropy boundaries from tune set
        tune_entropy = calculate_entropy(logits_tune)
        self.entropy_boundaries = self._get_entropy_boundaries(tune_entropy)
        
        # Assign calibration data to strata
        cal_entropy = calculate_entropy(self.logits_cal)
        cal_strata = self._assign_to_strata(cal_entropy)
        
        # Compute separate quantile for each stratum
        self.stratum_quantiles = {}
        self.stratum_sizes = {}
        
        for s in range(n_strata):
            stratum_mask = (cal_strata == s)
            n_s = stratum_mask.sum()
            
            if n_s == 0:
                print(f"      Warning: Stratum {s} has 0 samples!")
                self.stratum_quantiles[s] = 1.0
                self.stratum_sizes[s] = 0
                continue
            
            # Get APS scores for this stratum
            stratum_logits = self.logits_cal[stratum_mask]
            stratum_labels = self.labels_cal[stratum_mask]
            
            scores = self._compute_aps_scores(stratum_logits, stratum_labels)
            
            # Compute stratum-specific quantile
            q_level = np.ceil((n_s + 1) * (1 - self.alpha)) / n_s
            q_hat_s = np.quantile(scores, q_level, method='higher')
            
            self.stratum_quantiles[s] = q_hat_s
            self.stratum_sizes[s] = n_s.item()
            
            print(f"      Stratum {s}: n={n_s}, q_hat={q_hat_s:.4f}")
    
    def _get_entropy_boundaries(self, entropy_values):
        """Define stratum boundaries based on quantiles."""
        quantiles = np.linspace(0, 1, self.n_strata + 1)[1:-1]
        boundaries = np.quantile(entropy_values, quantiles)
        return boundaries
    
    def _assign_to_strata(self, entropy_values):
        """Assign samples to strata based on entropy."""
        strata = np.zeros(len(entropy_values), dtype=int)
        
        for i, ent in enumerate(entropy_values):
            stratum = 0
            for boundary in self.entropy_boundaries:
                if ent > boundary:
                    stratum += 1
                else:
                    break
            strata[i] = stratum
        
        return strata
    
    def _compute_aps_scores(self, logits, labels):
        """Compute APS conformity scores."""
        probs = torch.softmax(logits, dim=1)
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
        """Forward pass with stratum-specific quantiles."""
        with torch.no_grad():
            logits = self.model(x)
            
            # Compute entropy for test samples
            test_entropy = calculate_entropy(logits)
            test_strata = self._assign_to_strata(test_entropy)
            
            # Compute probabilities
            probs = torch.softmax(logits, dim=1)
            sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            
            for i in range(x.size(0)):
                # Get stratum-specific quantile
                stratum = test_strata[i]
                q_hat = self.stratum_quantiles[stratum]
                
                # Build prediction set
                mask = cumsum[i] <= q_hat
                
                if mask.sum() == 0:
                    k_star = 0
                else:
                    k_star = mask.nonzero(as_tuple=True)[0][-1].item()
                
                pred_set = idx[i, :k_star+1].cpu().numpy()
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets


def predict_with_sets(conformal_model, dataloader):
    """
    Mondrian (Group-Conditional) Conformal Prediction.
    
    Uses entropy-defined strata with SEPARATE quantile calibration per group.
    This provides theoretical guarantee of per-stratum validity under 
    exchangeability, at the cost of potential efficiency loss on easy strata.
    """
    
    def __init__(self, model, tune_loader, calib_loader, val_loader, alpha=0.1, 
                 n_strata=3, device='cuda'):
        """
        Args:
            model: Trained classifier
            tune_loader: To define entropy boundaries
            calib_loader: To compute per-stratum quantiles
            val_loader: Unused (Mondrian doesn't need selection)
            alpha: Miscoverage level
            n_strata: Number of entropy strata
            device: cuda or cpu
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.n_strata = n_strata
        self.device = device
        self.num_classes = model.num_classes
        
        print(f"   [Mondrian-CP] Calibrating with {n_strata} entropy strata...")
        
        # Get logits
        logits_tune, _ = get_logits_labels(model, tune_loader, device)
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        
        # Define entropy boundaries based on tuning set
        tune_entropy = calculate_entropy(logits_tune)
        self.entropy_boundaries = self._get_entropy_boundaries(tune_entropy)
        
        # Assign calibration data to strata
        cal_entropy = calculate_entropy(self.logits_cal)
        cal_strata = self._assign_to_strata(cal_entropy)
        
        # Compute separate quantile for each stratum
        self.stratum_quantiles = {}
        self.stratum_sizes = {}
        
        for s in range(n_strata):
            stratum_mask = (cal_strata == s)
            n_s = stratum_mask.sum()
            
            if n_s == 0:
                print(f"      Warning: Stratum {s} has 0 samples in calibration!")
                self.stratum_quantiles[s] = 1.0
                self.stratum_sizes[s] = 0
                continue
            
            stratum_logits = self.logits_cal[stratum_mask]
            stratum_labels = self.labels_cal[stratum_mask]
            
            scores = self._compute_aps_scores(stratum_logits, stratum_labels)
            
            q_level = np.ceil((n_s + 1) * (1 - alpha)) / n_s
            q_hat_s = np.quantile(scores, q_level, method='higher')
            
            self.stratum_quantiles[s] = q_hat_s
            self.stratum_sizes[s] = n_s.item()
            
            print(f"      Stratum {s}: n={n_s}, q_hat={q_hat_s:.4f}")
    
    def _calculate_entropy(self, logits):
        """Compute predictive entropy."""
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return entropy(probs, axis=1)
    
    def _get_entropy_boundaries(self, entropy_values):
        """Define stratum boundaries based on quantiles."""
        quantiles = np.linspace(0, 1, self.n_strata + 1)[1:-1]
        boundaries = np.quantile(entropy_values, quantiles)
        return boundaries
    
    def _assign_to_strata(self, entropy_values):
        """Assign samples to strata based on entropy."""
        strata = np.zeros(len(entropy_values), dtype=int)
        
        for i, ent in enumerate(entropy_values):
            stratum = 0
            for boundary in self.entropy_boundaries:
                if ent > boundary:
                    stratum += 1
                else:
                    break
            strata[i] = stratum
        
        return strata
    
    def _compute_aps_scores(self, logits, labels):
        """Compute APS conformity scores."""
        probs = torch.softmax(logits, dim=1)
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
        """Forward pass with stratum-specific quantiles."""
        with torch.no_grad():
            logits = self.model(x)
            
            test_entropy = self._calculate_entropy(logits)
            test_strata = self._assign_to_strata(test_entropy)
            
            probs = torch.softmax(logits, dim=1)
            sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            
            for i in range(x.size(0)):
                stratum = test_strata[i]
                q_hat = self.stratum_quantiles[stratum]
                
                mask = cumsum[i] <= q_hat
                
                if mask.sum() == 0:
                    k_star = 0
                else:
                    k_star = mask.nonzero(as_tuple=True)[0][-1].item()
                
                pred_set = idx[i, :k_star+1].cpu().numpy()
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets


def predict_with_sets(conformal_model, dataloader):
    """
    Helper function for evaluation.
    Required by evaluator.py.
    """
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

    return (np.concatenate(all_logits), 
            np.concatenate(all_preds), 
            np.concatenate(all_probs), 
            all_sets, 
            np.concatenate(all_labels))
