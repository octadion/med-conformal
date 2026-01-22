import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from typing import List, Tuple, Dict

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
    probs = torch.softmax(logits, dim=1).numpy()
    return entropy(probs, axis=1)
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
    def __init__(self, model, calib_loader, alpha=0.1, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.device = device
        
        print("   [APS] Calibrating...")
        logits, labels = get_logits_labels(model, calib_loader, device)
        
        scores = self._compute_aps_scores(logits, labels)
        
        n = len(labels)
        self.q_hat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
        print(f"   [APS] Q_hat: {self.q_hat:.4f}")

    def _compute_aps_scores(self, logits, labels=None):
        probs = torch.softmax(logits, dim=1)
        sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        
        if labels is None:
            return cumsum, idx
            
        # Score = Cumsum probability up to true class
        rows = torch.arange(len(labels))
        ranks = torch.zeros(len(labels), dtype=torch.long)
        for i, lbl in enumerate(labels):
            rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
            ranks[i] = rank.item() if len(rank) > 0 else 0
            
        scores = cumsum[rows, ranks]
        return scores.numpy()

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            cumsum, idx = self._compute_aps_scores(logits)
            
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


class SizeOptimizedRAPS(nn.Module):
    def __init__(self, model, calib_loader, tune_loader, alpha=0.1, k_reg=2, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.device = device
        self.num_classes = model.num_classes
        
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        
        print("   [RAPS-Standard] Optimizing Lambda for Minimal Set Size...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_size()
        print(f"   [RAPS-Standard] Lambda: {self.lamda_star:.5f}")

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

    def _optimize_lambda_size(self):
        lambda_grid = np.linspace(0, 0.05, 20)
        best_lam, best_q = 0.0, 0.0
        min_size = float('inf')
        target_cov = 1 - self.alpha
        
        for lam in lambda_grid:
            q_cand = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            probs_tune = torch.softmax(self.logits_tune, dim=1)
            sorted_probs, idx = torch.sort(probs_tune, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            penalties = lam * torch.clamp(torch.arange(self.num_classes) - self.k_reg + 1, min=0)
            scores_all = cumsum + penalties.unsqueeze(0)
            sizes = (scores_all <= q_cand).sum(dim=1)
            sizes = torch.clamp(sizes, min=1)
            avg_size = sizes.float().mean().item()
            
            rows = torch.arange(len(self.labels_tune))
            ranks = torch.zeros(len(self.labels_tune), dtype=torch.long)
            for i, lbl in enumerate(self.labels_tune):
                 rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
                 ranks[i] = rank.item() if len(rank) > 0 else 0
            
            scores_true = cumsum[rows, ranks] + lam * torch.clamp(ranks - self.k_reg + 1, min=0)
            coverage = (scores_true <= q_cand).float().mean().item()
            
            if coverage >= target_cov - 0.005: 
                if avg_size < min_size:
                    min_size = avg_size
                    best_lam = lam
                    best_q = q_cand
        
        return best_lam, best_q

    def forward(self, x):
        return _forward_raps(self, x)

class EntropyStratifiedRAPS(nn.Module):
    def __init__(self, model, calib_loader, tune_loader, alpha=0.1, k_reg=2, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.k_reg = k_reg
        self.num_classes = model.num_classes
        
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)
        
        print("   [Entropy-RAPS] Optimizing Lambda for Stratified Safety...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"   [Entropy-RAPS] Lambda: {self.lamda_star:.5f}")

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
        th1, th2 = np.quantile(tune_entropy, [0.33, 0.66])
        
        strata_idxs = [
            np.where(tune_entropy <= th1)[0],
            np.where((tune_entropy > th1) & (tune_entropy <= th2))[0],
            np.where(tune_entropy > th2)[0]
        ]
        
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

def _forward_raps(self, x):
    with torch.no_grad():
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        
        prediction_sets = []
        for i in range(x.size(0)):
            penalties = self.lamda_star * torch.clamp(torch.arange(self.num_classes, device=x.device) - self.k_reg + 1, min=0)
            scores = cumsum[i] + penalties
            
            mask = scores <= self.q_hat_star
            if mask.sum() == 0: k_star = 0
            else: k_star = mask.nonzero()[-1].item()
                
            pred_set = idx[i, :k_star+1].cpu().numpy()
            prediction_sets.append(pred_set)
            
        return logits, prediction_sets
    
class MondrianCP(nn.Module):
    """
    Mondrian (Group-Conditional) Conformal Prediction.
    
    Uses entropy-defined strata with SEPARATE quantile calibration per group.
    This provides theoretical guarantee of per-stratum validity under 
    exchangeability, at the cost of potential efficiency loss on easy strata.
    
    Args:
        model: Trained classifier
        calib_loader: Calibration data loader
        tune_loader: Tuning set (to define entropy boundaries)
        alpha: Miscoverage level (default 0.1 for 90% coverage)
        n_strata: Number of entropy strata (default 3 for tertiles)
        device: 'cuda' or 'cpu'
    """
    
    def __init__(self, model, calib_loader, tune_loader, alpha=0.1, 
                 n_strata=3, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.n_strata = n_strata
        self.device = device
        self.num_classes = model.num_classes
        
        # Get logits and labels for calibration
        print(f"   [Mondrian-CP] Calibrating with {n_strata} entropy strata...")
        self.logits_cal, self.labels_cal = self._get_logits_labels(calib_loader)
        logits_tune, _ = self._get_logits_labels(tune_loader)
        
        # Define entropy boundaries based on tuning set
        tune_entropy = self._calculate_entropy(logits_tune)
        self.entropy_boundaries = self._get_entropy_boundaries(tune_entropy)
        
        # Assign calibration data to strata
        cal_entropy = self._calculate_entropy(self.logits_cal)
        cal_strata = self._assign_to_strata(cal_entropy)
        
        # Compute separate quantile for each stratum
        self.stratum_quantiles = {}
        self.stratum_sizes = {}
        
        for s in range(n_strata):
            stratum_mask = (cal_strata == s)
            n_s = stratum_mask.sum()
            
            if n_s == 0:
                print(f"      Warning: Stratum {s} has 0 samples in calibration!")
                self.stratum_quantiles[s] = 1.0  # Conservative fallback
                self.stratum_sizes[s] = 0
                continue
            
            # Get APS scores for this stratum
            stratum_logits = self.logits_cal[stratum_mask]
            stratum_labels = self.labels_cal[stratum_mask]
            
            scores = self._compute_aps_scores(stratum_logits, stratum_labels)
            
            # Compute stratum-specific quantile
            q_level = np.ceil((n_s + 1) * (1 - alpha)) / n_s
            q_hat_s = np.quantile(scores, q_level, method='higher')
            
            self.stratum_quantiles[s] = q_hat_s
            self.stratum_sizes[s] = n_s.item()
            
            print(f"      Stratum {s}: n={n_s}, q_hat={q_hat_s:.4f}")
    
    def _get_logits_labels(self, loader):
        """Extract logits and labels from dataloader."""
        self.model.eval()
        logits_list, labels_list = [], []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                out = self.model(x)
                logits_list.append(out.cpu())
                
                if len(y.shape) > 1:
                    y = y.squeeze()
                labels_list.append(y.cpu())
        
        return torch.cat(logits_list), torch.cat(labels_list)
    
    def _calculate_entropy(self, logits):
        """Compute predictive entropy."""
        probs = torch.softmax(logits, dim=1).numpy()
        return entropy(probs, axis=1)
    
    def _get_entropy_boundaries(self, entropy_values):
        """Define stratum boundaries based on quantiles."""
        quantiles = np.linspace(0, 1, self.n_strata + 1)[1:-1]  # Exclude 0 and 1
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
        
        # Find rank of true label
        rows = torch.arange(len(labels))
        ranks = torch.zeros(len(labels), dtype=torch.long)
        
        for i, lbl in enumerate(labels):
            rank = (idx[i] == lbl).nonzero(as_tuple=True)[0]
            ranks[i] = rank.item() if len(rank) > 0 else 0
        
        scores = cumsum[rows, ranks]
        return scores.numpy()
    
    def forward(self, x):
        """
        Forward pass: predict with stratum-specific quantiles.
        
        Returns:
            logits: Model outputs
            prediction_sets: List of numpy arrays (one per sample)
        """
        with torch.no_grad():
            logits = self.model(x)
            
            # Compute entropy for test samples
            test_entropy = self._calculate_entropy(logits)
            test_strata = self._assign_to_strata(test_entropy)
            
            # Compute probabilities and cumulative sums
            probs = torch.softmax(logits, dim=1)
            sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            
            for i in range(x.size(0)):
                # Get stratum-specific quantile
                stratum = test_strata[i]
                q_hat = self.stratum_quantiles[stratum]
                
                # Build prediction set (include until cumsum exceeds q_hat)
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
    REQUIRED by evaluator.py.
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
            if len(y.shape) > 1: y = y.squeeze()
            all_labels.append(y.cpu().numpy())
            all_sets.extend(sets)

    return np.concatenate(all_logits), np.concatenate(all_preds), np.concatenate(all_probs), all_sets, np.concatenate(all_labels)