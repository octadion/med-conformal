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
            labels_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(labels_list).squeeze()

def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=1).numpy()
    return entropy(probs, axis=1)

class StandardLAC(nn.Module):
    def __init__(self, model, calib_loader, alpha=0.1, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        
        print("Calibrating Standard LAC...")
        logits, labels = get_logits_labels(model, calib_loader, device)
        probs = torch.softmax(logits, dim=1)
        
        true_probs = probs[torch.arange(len(labels)), labels.long()]
        scores = 1 - true_probs

        n = len(labels)
        q_val = np.quantile(scores.numpy(), np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
        
        self.q_hat = q_val
        print(f"LAC Threshold (1 - q_hat): {1 - self.q_hat:.4f}")

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            
            # if prob >= 1 - q_hat
            threshold = 1 - self.q_hat
            
            prediction_sets = []
            for p in probs.cpu().numpy():
                # Get indices where prob >= threshold
                pred_set = np.where(p >= threshold)[0]
                if len(pred_set) == 0: 
                     pred_set = np.array([np.argmax(p)])
                prediction_sets.append(pred_set)
                
            return logits, prediction_sets

class EntropyStratifiedRAPS(nn.Module):
    def __init__(self, model, calib_loader, tune_loader, alpha=0.1, k_reg=2, device='cuda'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.k_reg = k_reg
        self.device = device
        self.num_classes = model.num_classes
        
        print("Loading Calibration & Tuning Data...")
        self.logits_cal, self.labels_cal = get_logits_labels(model, calib_loader, device)
        self.logits_tune, self.labels_tune = get_logits_labels(model, tune_loader, device)

        print("Optimizing Lambda via Entropy Stratification...")
        self.lamda_star, self.q_hat_star = self._optimize_lambda_entropy()
        print(f"✓ Selected Lambda: {self.lamda_star:.5f} | Q_hat: {self.q_hat_star:.5f}")

    def _get_qhat(self, logits, labels, lamda):
        probs = torch.softmax(logits, dim=1)
        # Sort probabilities descending
        sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        
        rows = torch.arange(len(labels))
        ranks = torch.zeros(len(labels), dtype=torch.long)
        for i, lbl in enumerate(labels):
            ranks[i] = (idx[i] == lbl).nonzero(as_tuple=True)[0].item()
            
        # RAPS Score: Cumsum(true_class) + Lambda * Penalty
        # Penalty = max(0, rank - k_reg + 1)
        scores = cumsum[rows, ranks] + lamda * torch.clamp(ranks - self.k_reg + 1, min=0)
        
        n = len(labels)
        q = np.quantile(scores.numpy(), np.ceil((n + 1) * (1 - self.alpha)) / n, method='higher')
        return q

    def _optimize_lambda_entropy(self):
        tune_entropy = calculate_entropy(self.logits_tune)

        th1 = np.quantile(tune_entropy, 0.33)
        th2 = np.quantile(tune_entropy, 0.66)
        
        idxs_easy = np.where(tune_entropy <= th1)[0]
        idxs_med  = np.where((tune_entropy > th1) & (tune_entropy <= th2))[0]
        idxs_hard = np.where(tune_entropy > th2)[0]
        
        strata_idxs = [idxs_easy, idxs_med, idxs_hard]
        # lambda_grid = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        lambda_grid = np.linspace(0, 0.05, 20) # Fine-grained search
        
        best_lam = 0.0
        best_q = 0.0
        min_max_violation = float('inf')
        
        for lam in lambda_grid:
            q_candidate = self._get_qhat(self.logits_cal, self.labels_cal, lam)
            
            probs_tune = torch.softmax(self.logits_tune, dim=1)
            sorted_probs, idx = torch.sort(probs_tune, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            covered = []
            # Score(label) = Cumsum(label) + Penalty(label)
            
            rows = torch.arange(len(self.labels_tune))
            ranks = torch.zeros(len(self.labels_tune), dtype=torch.long)
            for i, lbl in enumerate(self.labels_tune):
                ranks[i] = (idx[i] == lbl).nonzero(as_tuple=True)[0].item()
            
            scores_tune = cumsum[rows, ranks] + lam * torch.clamp(ranks - self.k_reg + 1, min=0)
            is_covered = (scores_tune <= q_candidate).numpy().astype(int)
            
            violations = []
            for stratum_idx in strata_idxs:
                if len(stratum_idx) == 0: continue
                cov_strat = np.mean(is_covered[stratum_idx])
                violations.append(abs(cov_strat - (1 - self.alpha)))
            
            max_viol = max(violations) if violations else 1.0
            
            if max_viol < min_max_violation:
                min_max_violation = max_viol
                best_lam = lam
                best_q = q_candidate
                
        return best_lam, best_q

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            sorted_probs, idx = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            
            prediction_sets = []
            for i in range(x.size(0)):
                # Construct Set: Include k until score > Q_hat
                # Optimization: Find cut-off k
                # Score[k] = cumsum[k] + lambda * max(0, k - kreg + 1)
                
                penalties = self.lamda_star * torch.clamp(torch.arange(self.num_classes, device=x.device) - self.k_reg + 1, min=0)
                scores = cumsum[i] + penalties
                
                mask = scores <= self.q_hat_star
                
                if mask.sum() == 0:
                    k_star = 0 
                else:
                    k_star = mask.nonzero()[-1].item()
                    
                pred_set = idx[i, :k_star+1].cpu().numpy()
                prediction_sets.append(pred_set)
                
            return logits, prediction_sets