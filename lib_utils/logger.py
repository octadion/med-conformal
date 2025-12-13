import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

def convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class ExperimentLogger:
    def __init__(self, log_dir: str, experiment_name: str, console_level: str = "INFO"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            console_level: Logging level for console output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        (self.log_dir / "logs").mkdir(exist_ok=True)
        (self.log_dir / "results").mkdir(exist_ok=True)
        (self.log_dir / "models").mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{self.timestamp}"

        log_file = self.log_dir / "logs" / f"{self.experiment_name}.log"
        
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.metrics_history = {
            'train': [],
            'val': [],
            'test': {}
        }
        
        self.info(f"Logger initialized. Logs saved to: {log_file}")
        
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def log_epoch(self, epoch: int, phase: str, metrics: Dict[str, float]):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            phase: 'train' or 'val'
            metrics: Dictionary of metrics
        """
        metrics_copy = metrics.copy()
        metrics_copy['epoch'] = epoch
        
        if phase in ['train', 'val']:
            self.metrics_history[phase].append(metrics_copy)
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch} [{phase.upper()}] - {metric_str}")
        
    def log_test_results(self, method: str, metrics: Dict[str, Any]):
        """
        Log test results for a method.
        
        Args:
            method: Method name (e.g., 'RAPS_Adaptive')
            metrics: Dictionary of test metrics
        """
        self.metrics_history['test'][method] = metrics
        
        self.info(f"\n{'='*60}")
        self.info(f"Test Results - {method}")
        self.info(f"{'='*60}")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        self.info(f"{'='*60}\n")
        
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_file = self.log_dir / "results" / f"{self.experiment_name}_metrics.json"

        serializable_metrics = convert_to_serializable(self.metrics_history)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        self.info(f"Metrics saved to: {metrics_file}")
        
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file."""
        config_file = self.log_dir / "results" / f"{self.experiment_name}_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        self.info(f"Config saved to: {config_file}")
        
    def get_model_save_path(self, name: str = "best_model.pth") -> Path:
        """Get path to save model."""
        return self.log_dir / "models" / f"{self.experiment_name}_{name}"
    
    def get_results_dir(self) -> Path:
        """Get results directory."""
        results_dir = self.log_dir / "results" / self.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def create_summary(self):
        """Create a summary of all results."""
        summary_file = self.log_dir / "results" / f"{self.experiment_name}_SUMMARY.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"EXPERIMENT SUMMARY: {self.experiment_name}\n")
            f.write("="*80 + "\n\n")

            if self.metrics_history['train']:
                best_train = max(self.metrics_history['train'], 
                               key=lambda x: x.get('accuracy', 0))
                f.write("TRAINING:\n")
                f.write(f"  Best Epoch: {best_train['epoch']}\n")
                f.write(f"  Best Accuracy: {best_train.get('accuracy', 0):.4f}\n\n")

            if self.metrics_history['val']:
                best_val = max(self.metrics_history['val'], 
                             key=lambda x: x.get('accuracy', 0))
                f.write("VALIDATION:\n")
                f.write(f"  Best Epoch: {best_val['epoch']}\n")
                f.write(f"  Best Accuracy: {best_val.get('accuracy', 0):.4f}\n\n")

            if self.metrics_history['test']:
                f.write("TEST RESULTS:\n")
                f.write("-"*80 + "\n\n")
                for method, metrics in self.metrics_history['test'].items():
                    f.write(f"{method}:\n")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {key}: {value:.4f}\n")
                    f.write("\n")
        
        self.info(f"Summary saved to: {summary_file}")


class MetricsTracker:
    """metrics tracker for training."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
        
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key: str) -> float:
        """Get average of a metric."""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return sum(self.metrics[key]) / len(self.metrics[key])
        return 0.0
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all average metrics."""
        return {key: self.get_average(key) for key in self.metrics.keys()}
