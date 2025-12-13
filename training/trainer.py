import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
import sys
sys.path.append('.')

from lib_utils.logger import MetricsTracker
from lib_utils.metrics import compute_classification_metrics


class Trainer:
    """Trainer for classification model."""
    
    def __init__(self, model: nn.Module, config: dict, device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device
        self.train_config = config['training']
        
        label_smoothing = self.train_config.get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = self._get_optimizer()
        
        self.scheduler = self._get_scheduler()
        
        self.early_stopping = EarlyStopping(
            patience=self.train_config['early_stopping']['patience'],
            min_delta=self.train_config['early_stopping']['min_delta']
        )

        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on config."""
        optimizer_name = self.train_config['optimizer'].lower()
        lr = self.train_config['learning_rate']
        weight_decay = self.train_config.get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler based on config."""
        scheduler_name = self.train_config.get('scheduler', 'none').lower()
        
        if scheduler_name == 'none':
            return None
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config['epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device).squeeze()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(y).sum().item()
            accuracy = correct / y.size(0)
            
            # Update metrics
            metrics_tracker.update(
                loss=loss.item(),
                accuracy=accuracy
            )
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}"
            })
        
        metrics = metrics_tracker.get_all_averages()
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device).squeeze()
            
            # Forward pass
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store for metrics
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Compute accuracy
            correct = predicted.eq(y).sum().item()
            accuracy = correct / y.size(0)
            
            # Update metrics
            metrics_tracker.update(
                loss=loss.item(),
                accuracy=accuracy
            )
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}"
            })
        
        # Concatenate all predictions
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        clf_metrics = compute_classification_metrics(
            all_labels, all_preds, all_probs,
            num_classes=self.model.num_classes
        )
        
        metrics = metrics_tracker.get_all_averages()
        metrics.update({
            'precision': clf_metrics['precision'],
            'recall': clf_metrics['recall'],
            'f1_score': clf_metrics['f1_score']
        })
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             logger) -> nn.Module:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            logger: Logger instance
            
        Returns:
            Trained model
        """
        num_epochs = self.train_config['epochs']
        
        logger.info(f"\nStarting training for {num_epochs} epochs...")
        logger.info(f"Optimizer: {self.train_config['optimizer']}")
        logger.info(f"Learning rate: {self.train_config['learning_rate']}")
        logger.info(f"Batch size: {self.config['data']['batch_size']}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.log_epoch(epoch, 'train', train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            logger.log_epoch(epoch, 'val', val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logger.debug(f"Learning rate: {current_lr:.6f}")
            
            # Save best model
            val_acc = val_metrics['accuracy']
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                logger.info(f"  → New best validation accuracy: {val_acc:.4f}")
                
                # Save checkpoint
                save_path = logger.get_model_save_path()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics
                }, save_path)
            
            # Early stopping check
            if self.train_config['early_stopping']['enabled']:
                self.early_stopping.update(val_acc)
                if self.early_stopping.should_stop():
                    logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"\nLoaded best model with validation accuracy: {self.best_val_acc:.4f}")
        
        return self.model


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def update(self, score: float):
        """
        Update early stopping with new score.
        
        Args:
            score: Current score (higher is better)
        """
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.counter >= self.patience
