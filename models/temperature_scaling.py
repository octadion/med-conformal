"""
Temperature Scaling for Conformal Prediction (FIXED)

FIXED: Uses y.view(-1) instead of y.squeeze() to handle single-sample batches
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


class TemperatureScaler:
    """
    Temperature Scaling (Platt Scaling) for probability calibration.
    """
    
    def __init__(self, model: nn.Module, calib_loader, 
                 device: str = 'cuda',
                 max_iters: int = 50,
                 lr: float = 0.01,
                 epsilon: float = 0.01,
                 verbose: bool = True):
        self.model = model
        self.device = device
        self.verbose = verbose
        
        logits, labels = self._get_logits_labels(calib_loader)
        self.temperature = self._optimize_temperature(
            logits, labels, max_iters, lr, epsilon
        )
        
        if verbose:
            print(f"   [Temperature Scaling] Optimal T = {self.temperature:.4f}")
    
    def _get_logits_labels(self, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract logits and labels from dataloader."""
        self.model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                out = self.model(x)
                logits_list.append(out.cpu())
                # FIX: Use view(-1) instead of squeeze() to handle single-sample batches
                y = y.cpu().view(-1)
                labels_list.append(y)
        
        return torch.cat(logits_list), torch.cat(labels_list)
    
    def _optimize_temperature(self, logits: torch.Tensor, 
                             labels: torch.Tensor,
                             max_iters: int,
                             lr: float,
                             epsilon: float) -> float:
        temperature = nn.Parameter(torch.ones(1).to(self.device) * 1.3)
        optimizer = optim.SGD([temperature], lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        logits = logits.to(self.device)
        labels = labels.long().to(self.device)
        
        prev_temp = temperature.item()
        for iteration in range(max_iters):
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            optimizer.step()
            
            if iteration > 0:
                delta = abs(temperature.item() - prev_temp)
                if delta < epsilon:
                    if self.verbose:
                        print(f"      Converged at iteration {iteration+1} (ΔT={delta:.6f})")
                    break
            
            prev_temp = temperature.item()
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"      Iter {iteration+1}/{max_iters}: T={temperature.item():.4f}, Loss={loss.item():.4f}")
        
        return temperature.item()
    
    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
    
    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_logits = self.scale_logits(logits)
        return torch.softmax(scaled_logits, dim=1)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate_calibration(model: nn.Module, 
                        dataloader,
                        temperature: Optional[float] = None,
                        device: str = 'cuda') -> dict:
    """Evaluate calibration before and after temperature scaling."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            
            if temperature is not None:
                logits = logits / temperature
            
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            # FIX: Use view(-1) instead of squeeze()
            y = y.cpu().view(-1).numpy()
            all_labels.append(y)
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    predictions = np.argmax(all_probs, axis=1)
    accuracy = np.mean(predictions == all_labels)
    ece = compute_ece(all_probs, all_labels)
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'temperature': temperature if temperature is not None else 1.0
    }