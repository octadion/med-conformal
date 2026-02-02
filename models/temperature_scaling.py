"""
Temperature Scaling for Conformal Prediction

This module implements temperature scaling (Platt scaling) to improve
probability calibration before applying conformal prediction methods.

Based on:
- Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
- Original conformal.py implementation

Usage:
    # Calibrate temperature
    scaler = TemperatureScaler(model, calib_loader, device='cuda')
    T_optimal = scaler.temperature
    
    # Apply to conformal method
    ts_aps = TemperatureScaledAPS(model, calib_loader, T=T_optimal, ...)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


class TemperatureScaler:
    """
    Temperature Scaling (Platt Scaling) for probability calibration.
    
    Learns a single scalar parameter T that divides logits before softmax:
        p_calibrated = softmax(logits / T)
    
    This improves calibration without changing the model's predictions.
    """
    
    def __init__(self, model: nn.Module, calib_loader, 
                 device: str = 'cuda',
                 max_iters: int = 50,
                 lr: float = 0.01,
                 epsilon: float = 0.01,
                 verbose: bool = True):
        """
        Calibrate temperature parameter via NLL minimization.
        
        Args:
            model: Trained classifier
            calib_loader: DataLoader for calibration
            device: 'cuda' or 'cpu'
            max_iters: Maximum optimization iterations
            lr: Learning rate for SGD
            epsilon: Convergence threshold
            verbose: Print optimization progress
        """
        self.model = model
        self.device = device
        self.verbose = verbose
        
        # Extract logits and labels from calibration set
        logits, labels = self._get_logits_labels(calib_loader)
        
        # Optimize temperature
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
                
                if len(y.shape) > 1:
                    y = y.squeeze()
                labels_list.append(y.cpu())
        
        return torch.cat(logits_list), torch.cat(labels_list)
    
    def _optimize_temperature(self, logits: torch.Tensor, 
                             labels: torch.Tensor,
                             max_iters: int,
                             lr: float,
                             epsilon: float) -> float:
        """
        Optimize temperature parameter via NLL minimization.
        
        Args:
            logits: Model outputs (N, num_classes)
            labels: Ground truth labels (N,)
            max_iters: Maximum iterations
            lr: Learning rate
            epsilon: Convergence threshold
            
        Returns:
            Optimal temperature value
        """
        # Initialize temperature parameter
        temperature = nn.Parameter(torch.ones(1).to(self.device) * 1.3)
        
        # Setup optimizer and loss
        optimizer = optim.SGD([temperature], lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Move data to device
        logits = logits.to(self.device)
        labels = labels.long().to(self.device)
        
        # Optimization loop
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            # Scale logits by temperature
            scaled_logits = logits / temperature
            
            # Compute NLL loss
            loss = criterion(scaled_logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Check convergence
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
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model outputs (N, num_classes)
            
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get calibrated probabilities from logits.
        
        Args:
            logits: Model outputs (N, num_classes)
            
        Returns:
            Calibrated probabilities via softmax(logits/T)
        """
        scaled_logits = self.scale_logits(logits)
        return torch.softmax(scaled_logits, dim=1)


def compute_ece(probs: np.ndarray, labels: np.ndarray, 
                n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: Ground truth labels (N,)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better)
    """
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
    """
    Evaluate calibration before and after temperature scaling.
    
    Args:
        model: Trained classifier
        dataloader: DataLoader for evaluation
        temperature: Temperature parameter (None = no scaling)
        device: 'cuda' or 'cpu'
        
    Returns:
        Dictionary with ECE and accuracy metrics
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            
            # Apply temperature scaling if provided
            if temperature is not None:
                logits = logits / temperature
            
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
            if len(y.shape) > 1:
                y = y.squeeze()
            all_labels.append(y.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    predictions = np.argmax(all_probs, axis=1)
    accuracy = np.mean(predictions == all_labels)
    ece = compute_ece(all_probs, all_labels)
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'temperature': temperature if temperature is not None else 1.0
    }


# ============================================================================
# DEMONSTRATION / TESTING
# ============================================================================

if __name__ == "__main__":
    print("Temperature Scaling Module")
    print("=" * 60)
    print("\nThis module provides temperature scaling for conformal prediction.")
    print("\nKey classes:")
    print("  - TemperatureScaler: Calibrate T via NLL minimization")
    print("  - compute_ece(): Measure calibration quality")
    print("  - evaluate_calibration(): Before/after comparison")
    print("\nUsage example:")
    print("""
    # Calibrate temperature
    scaler = TemperatureScaler(model, calib_loader)
    T_opt = scaler.temperature
    
    # Evaluate improvement
    before = evaluate_calibration(model, test_loader, temperature=None)
    after = evaluate_calibration(model, test_loader, temperature=T_opt)
    
    print(f"ECE before: {before['ece']:.4f}")
    print(f"ECE after:  {after['ece']:.4f}")
    """)
    print("=" * 60)
