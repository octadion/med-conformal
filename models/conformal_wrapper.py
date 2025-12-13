import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from typing import List, Tuple, Optional
import sys
sys.path.append('/mnt/user-data/uploads')

from utils import sort_sum
from conformal import (
    ConformalModelLogits, platt_logits, 
    conformal_calibration_logits, pick_parameters
)


class BaseConformalWrapper(nn.Module):
    """Base class for conformal prediction wrappers."""
    
    def __init__(self, model: nn.Module, alpha: float = 0.1):
        """
        Initialize conformal wrapper.
        
        Args:
            model: Base classification model
            alpha: Significance level (1 - coverage)
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.num_classes = model.num_classes
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Forward pass returning logits and prediction sets.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (logits, prediction_sets)
        """
        raise NotImplementedError


class NaiveConformal(BaseConformalWrapper):
    """Naive baseline: predict class with confidence > threshold."""
    
    def __init__(self, model: nn.Module, alpha: float = 0.1):
        super().__init__(model, alpha)
        self.threshold = 1 - alpha
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Forward with naive prediction sets."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            
            prediction_sets = []
            for prob in probs.cpu().numpy():
                pred_set = np.where(prob >= self.threshold)[0]
                if len(pred_set) == 0:
                    pred_set = np.array([np.argmax(prob)])
                prediction_sets.append(pred_set)
            
            return logits, prediction_sets


class LACConformal(BaseConformalWrapper):
    """Label-conditional Adaptive Conformal (LAC) prediction."""
    
    def __init__(self, model: nn.Module, calib_loader, alpha: float = 0.1):
        super().__init__(model, alpha)
        
        self.calib_logits = self._get_logits_dataset(calib_loader)

        calib_loader_logits = torch.utils.data.DataLoader(
            self.calib_logits, batch_size=128, shuffle=False
        )
        
        self.conformal_model = ConformalModelLogits(
            model=model,
            calib_loader=calib_loader_logits,
            alpha=alpha,
            naive=False,
            LAC=True
        )
        
    def _get_logits_dataset(self, dataloader):
        """Extract logits and labels from dataloader."""
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.cuda()
                logits = self.model(x)
                all_logits.append(logits.cpu())
                all_labels.append(y.squeeze())
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return torch.utils.data.TensorDataset(logits, labels.long())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Forward with LAC prediction sets."""
        with torch.no_grad():
            logits = self.model(x)
            _, pred_sets = self.conformal_model(logits)
            return logits, pred_sets


class RAPSConformal(BaseConformalWrapper):
    """Regularized Adaptive Prediction Sets (RAPS)."""
    
    def __init__(self, model: nn.Module, paramtune_loader, calib_loader,
                 alpha: float = 0.1, lamda_criterion: str = 'size',
                 temperature_scaling: bool = False):
        super().__init__(model, alpha)
        
        self.lamda_criterion = lamda_criterion
        self.temperature_scaling = temperature_scaling
        
        self.paramtune_logits = self._get_logits_dataset(paramtune_loader)
        self.calib_logits = self._get_logits_dataset(calib_loader)
        
        # Combine for parameter selection
        combined_logits = torch.utils.data.ConcatDataset([
            self.paramtune_logits, self.calib_logits
        ])
        
        combined_loader = torch.utils.data.DataLoader(
            combined_logits, batch_size=128, shuffle=False
        )
        
        # Create RAPS conformal model
        self.conformal_model = ConformalModelLogits(
            model=model,
            calib_loader=combined_loader,
            alpha=alpha,
            kreg=None,  # Auto-tune
            lamda=None,  # Auto-tune
            randomized=True,
            allow_zero_sets=False,
            naive=False,
            LAC=False,
            pct_paramtune=0.3,
            batch_size=128,
            lamda_criterion=lamda_criterion
        )
        
    def _get_logits_dataset(self, dataloader):
        """Extract logits and labels from dataloader."""
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.cuda()
                logits = self.model(x)
                all_logits.append(logits.cpu())
                all_labels.append(y.squeeze())
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return torch.utils.data.TensorDataset(logits, labels.long())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Forward with RAPS prediction sets."""
        with torch.no_grad():
            logits = self.model(x)
            _, pred_sets = self.conformal_model(logits)
            return logits, pred_sets


def create_conformal_models(base_model: nn.Module, 
                            paramtune_loader, 
                            calib_loader,
                            config: dict) -> dict:
    """
    Create all conformal prediction models for comparison.
    
    Args:
        base_model: Trained base model
        paramtune_loader: DataLoader for parameter tuning
        calib_loader: DataLoader for calibration
        config: Configuration dictionary
        
    Returns:
        Dictionary of conformal models
    """
    alpha = config['conformal']['alpha']
    methods_to_run = config['conformal'].get('methods', [
        'naive', 'lac', 'raps_size', 'raps_temp', 'raps_adaptive'
    ])
    
    models = {}
    
    print("\n" + "="*60)
    print("Creating Conformal Prediction Models")
    print("="*60)
    
    if 'naive' in methods_to_run:
        print("\n1. Naive (Softmax Threshold)")
        models['Naive'] = NaiveConformal(base_model, alpha=alpha)
        print("   ✓ Created")
    
    if 'lac' in methods_to_run:
        print("\n2. LAC (Label-conditional Adaptive Conformal)")
        models['LAC'] = LACConformal(base_model, calib_loader, alpha=alpha)
        print("   ✓ Created")
    
    if 'raps_size' in methods_to_run:
        print("\n3. RAPS (Size-based Lambda)")
        models['RAPS_Size'] = RAPSConformal(
            base_model, paramtune_loader, calib_loader,
            alpha=alpha, lamda_criterion='size', temperature_scaling=False
        )
        print("   ✓ Created")
    
    if 'raps_temp' in methods_to_run:
        print("\n4. RAPS + Temperature Scaling")
        models['RAPS_Temp'] = RAPSConformal(
            base_model, paramtune_loader, calib_loader,
            alpha=alpha, lamda_criterion='size', temperature_scaling=True
        )
        print("   ✓ Created")
    
    if 'raps_adaptive' in methods_to_run:
        print("\n5. RAPS + Adaptive (Main Method)")
        models['RAPS_Adaptive'] = RAPSConformal(
            base_model, paramtune_loader, calib_loader,
            alpha=alpha, lamda_criterion='adaptiveness', temperature_scaling=True
        )
        print("   ✓ Created")
    
    print("\n" + "="*60 + "\n")
    
    return models


def predict_with_sets(model: BaseConformalWrapper, dataloader) -> Tuple:
    """
    Get predictions and prediction sets for entire dataset.
    
    Args:
        model: Conformal model
        dataloader: DataLoader
        
    Returns:
        Tuple of (all_logits, all_preds, all_probs, all_sets, all_labels)
    """
    all_logits = []
    all_pred_sets = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            logits, pred_sets = model(x)
            
            all_logits.append(logits.cpu())
            all_pred_sets.extend(pred_sets)
            all_labels.append(y)
    
    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.softmax(all_logits, dim=1).numpy()
    all_preds = np.argmax(all_probs, axis=1)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    return all_logits, all_preds, all_probs, all_pred_sets, all_labels
