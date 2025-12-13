import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class Visualizer:
    """Create all visualizations for the experiment."""
    
    def __init__(self, save_dir: Path, config: dict):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
            config: Configuration dictionary
        """
        self.save_dir = Path(save_dir)
        self.config = config
        
        (self.save_dir / 'training').mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'conformal').mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'comparison').mkdir(parents=True, exist_ok=True)
        
    def plot_training_curves(self, train_history: List[Dict], 
                            val_history: List[Dict]):
        """Plot training and validation curves."""
        epochs = [h['epoch'] for h in train_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        train_loss = [h['loss'] for h in train_history]
        val_loss = [h['loss'] for h in val_history]
        
        ax1.plot(epochs, train_loss, label='Train', marker='o', markersize=3)
        ax1.plot(epochs, val_loss, label='Validation', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        train_acc = [h['accuracy'] for h in train_history]
        val_acc = [h['accuracy'] for h in val_history]
        
        ax2.plot(epochs, train_acc, label='Train', marker='o', markersize=3)
        ax2.plot(epochs, val_acc, label='Validation', marker='s', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training' / 'training_curves.png', 
                   bbox_inches='tight')
        plt.close()
        
        print("  ✓ Training curves saved")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             method_name: str = "Base Model"):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True Label',
               xlabel='Predicted Label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                       ha="center", va="center",
                       color="white" if cm_norm[i, j] > thresh else "black",
                       fontsize=8)
        
        ax.set_title(f'Confusion Matrix - {method_name}')
        plt.tight_layout()
        
        filename = method_name.lower().replace(' ', '_')
        plt.savefig(self.save_dir / 'training' / f'confusion_matrix_{filename}.png',
                   bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Confusion matrix saved ({method_name})")
    
    def plot_coverage_comparison(self, all_results: Dict[str, Dict]):
        """Plot coverage comparison across methods."""
        methods = []
        coverages = []
        violations = []
        target = 1 - self.config['conformal']['alpha']
        
        for method, results in all_results.items():
            if 'uncertainty' in results:
                methods.append(method)
                coverages.append(results['uncertainty']['coverage'])
                violations.append(abs(results['uncertainty']['coverage'] - target))
        
        if not methods:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(methods))
        bars = ax1.bar(x, coverages, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axhline(y=target, color='r', linestyle='--', label=f'Target ({target:.2f})')
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Coverage')
        ax1.set_title('Coverage Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, cov) in enumerate(zip(bars, coverages)):
            if cov >= target - 0.01:
                bar.set_color('lightgreen')
            else:
                bar.set_color('lightcoral')
        
        bars2 = ax2.bar(x, violations, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Coverage Violation')
        ax2.set_title('Coverage Violation (|Actual - Target|)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'conformal' / 'coverage_comparison.png',
                   bbox_inches='tight')
        plt.close()
        
        print("  ✓ Coverage comparison saved")
    
    def plot_set_size_distribution(self, all_results: Dict[str, Dict]):
        """Plot set size distributions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for method, results in all_results.items():
            if 'uncertainty' not in results or plot_idx >= 6:
                continue
                
            dist = results['uncertainty']['set_size_distribution']
            sizes = list(dist.keys())
            counts = list(dist.values())
            
            ax = axes[plot_idx]
            ax.bar(sizes, counts, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Set Size')
            ax.set_ylabel('Count')
            ax.set_title(f'{method}')
            ax.grid(True, alpha=0.3, axis='y')

            avg_size = results['uncertainty']['avg_set_size']
            ax.axvline(x=avg_size, color='r', linestyle='--', 
                      label=f'Avg: {avg_size:.2f}')
            ax.legend()
            
            plot_idx += 1
        
        for idx in range(plot_idx, 6):
            axes[idx].axis('off')
        
        plt.suptitle('Set Size Distributions', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'conformal' / 'set_size_distributions.png',
                   bbox_inches='tight')
        plt.close()
        
        print("  ✓ Set size distributions saved")
    
    def plot_coverage_vs_setsize(self, all_results: Dict[str, Dict]):
        """Plot coverage vs average set size trade-off."""
        methods = []
        coverages = []
        set_sizes = []
        
        for method, results in all_results.items():
            if 'uncertainty' in results:
                methods.append(method)
                coverages.append(results['uncertainty']['coverage'])
                set_sizes.append(results['uncertainty']['avg_set_size'])
        
        if not methods:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(set_sizes, coverages, s=200, alpha=0.6, 
                           c=range(len(methods)), cmap='viridis',
                           edgecolors='black', linewidth=2)
        
        for i, method in enumerate(methods):
            ax.annotate(method, (set_sizes[i], coverages[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, weight='bold')
        
        target = 1 - self.config['conformal']['alpha']
        ax.axhline(y=target, color='r', linestyle='--', linewidth=2,
                  label=f'Target Coverage ({target:.2f})')
        
        ax.set_xlabel('Average Set Size', fontsize=12)
        ax.set_ylabel('Coverage', fontsize=12)
        ax.set_title('Coverage vs Set Size Trade-off', fontsize=14, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'conformal' / 'coverage_vs_setsize.png',
                   bbox_inches='tight')
        plt.close()
        
        print("  ✓ Coverage vs set size plot saved")
    
    def plot_per_class_coverage(self, all_results: Dict[str, Dict], 
                                class_names: List[str]):
        """Plot per-class coverage heatmap."""
        methods = []
        coverage_matrix = []
        
        for method, results in all_results.items():
            if 'uncertainty' in results:
                methods.append(method)
                class_coverages = []
                cond_cov = results['uncertainty']['conditional_coverage']
                for i in range(len(class_names)):
                    class_coverages.append(cond_cov.get(i, np.nan))
                coverage_matrix.append(class_coverages)
        
        if not methods:
            return
        
        coverage_matrix = np.array(coverage_matrix)
        
        fig, ax = plt.subplots(figsize=(14, len(methods) * 0.8 + 2))
        
        im = ax.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto',
                      vmin=0.8, vmax=1.0)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage', rotation=270, labelpad=20)
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(methods)

        target = 1 - self.config['conformal']['alpha']
        for i in range(len(methods)):
            for j in range(len(class_names)):
                value = coverage_matrix[i, j]
                if not np.isnan(value):
                    color = 'white' if value > target else 'black'
                    text = ax.text(j, i, f'{value:.3f}',
                                 ha="center", va="center", color=color,
                                 fontsize=8)
        
        ax.set_title('Per-Class Coverage', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'conformal' / 'per_class_coverage.png',
                   bbox_inches='tight')
        plt.close()
        
        print("  ✓ Per-class coverage heatmap saved")
    
    def plot_stratified_coverage(self, all_results: Dict[str, Dict]):
        """Plot coverage stratified by set size."""
        methods = []
        strata_data = {
            'size_1': [],
            'size_2': [],
            'size_3': [],
            'size_4+': []
        }
        
        for method, results in all_results.items():
            if 'uncertainty' in results:
                methods.append(method)
                strat_cov = results['uncertainty']['stratified_coverage']
                for stratum in strata_data.keys():
                    val = strat_cov.get(stratum, np.nan)
                    strata_data[stratum].append(val)
        
        if not methods:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, (stratum, values) in enumerate(strata_data.items()):
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=stratum, alpha=0.8)
        
        target = 1 - self.config['conformal']['alpha']
        ax.axhline(y=target, color='r', linestyle='--', linewidth=2,
                  label=f'Target ({target:.2f})')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Coverage')
        ax.set_title('Stratified Coverage by Set Size')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'conformal' / 'stratified_coverage.png',
                   bbox_inches='tight')
        plt.close()
        
        print("  ✓ Stratified coverage plot saved")
    
    def create_all_plots(self, train_history: List[Dict],
                        val_history: List[Dict],
                        all_results: Dict[str, Dict],
                        class_names: List[str]):
        """Create all plots at once."""
        print("\nGenerating visualizations...")
        
        self.plot_training_curves(train_history, val_history)
        
        for method, results in all_results.items():
            if 'confusion_matrix' in results:
                self.plot_confusion_matrix(
                    results['confusion_matrix'],
                    class_names,
                    method
                )

        self.plot_coverage_comparison(all_results)
        self.plot_set_size_distribution(all_results)
        self.plot_coverage_vs_setsize(all_results)
        self.plot_per_class_coverage(all_results, class_names)
        self.plot_stratified_coverage(all_results)
        
        print("\n✓ All visualizations saved!")
