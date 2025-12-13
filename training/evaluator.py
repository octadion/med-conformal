import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict
import time
import sys
sys.path.append('.')

from lib_utils.metrics import compute_comprehensive_metrics
from models.conformal_wrapper import predict_with_sets


class ComprehensiveEvaluator:
    """Evaluate conformal prediction methods comprehensively."""
    
    def __init__(self, config: dict, class_names: list):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            class_names: List of class names
        """
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def evaluate_method(self, conformal_model, dataloader: DataLoader,
                       method_name: str, compute_gradcam: bool = False) -> Dict:
        """
        Evaluate a single conformal prediction method.
        
        Args:
            conformal_model: Conformal prediction model
            dataloader: Test data loader
            method_name: Name of the method
            compute_gradcam: Whether to compute Grad-CAM
            
        Returns:
            Dictionary of evaluation results
        """
        print(f"\nEvaluating: {method_name}")
        print("-" * 60)

        start_time = time.time()

        all_logits, all_preds, all_probs, all_sets, all_labels = predict_with_sets(
            conformal_model, dataloader
        )
        
        inference_time = time.time() - start_time

        results = compute_comprehensive_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_probs,
            prediction_sets=all_sets,
            class_names=self.class_names,
            num_classes=self.num_classes
        )

        results['timing'] = {
            'total_inference_time': inference_time,
            'per_sample_time': inference_time / len(all_labels),
            'throughput': len(all_labels) / inference_time
        }

        results['method_name'] = method_name
        results['num_samples'] = len(all_labels)
        results['target_coverage'] = 1 - self.config['conformal']['alpha']

        self._print_summary(results, method_name)
        
        return results
    
    def _print_summary(self, results: Dict, method_name: str):
        """Print evaluation summary."""
        print(f"\n{method_name} Results:")
        print("=" * 60)
        
        clf = results['classification']
        print(f"Classification:")
        print(f"  Accuracy:  {clf['accuracy']:.4f}")
        print(f"  Precision: {clf['precision']:.4f}")
        print(f"  Recall:    {clf['recall']:.4f}")
        print(f"  F1-score:  {clf['f1_score']:.4f}")
        if 'auc_macro' in clf:
            print(f"  AUC (macro): {clf['auc_macro']:.4f}")
        
        # Uncertainty quantification
        if 'uncertainty' in results:
            unc = results['uncertainty']
            print(f"\nUncertainty Quantification:")
            print(f"  Coverage:      {unc['coverage']:.4f} (target: {results['target_coverage']:.4f})")
            print(f"  Avg Set Size:  {unc['avg_set_size']:.4f}")
            print(f"  Singleton Rate: {unc['efficiency']['singleton_rate']:.4f}")
            
            # Coverage violation
            violation = abs(unc['coverage'] - results['target_coverage'])
            print(f"  Coverage Violation: {violation:.4f}")

        timing = results['timing']
        print(f"\nTiming:")
        print(f"  Total Time:    {timing['total_inference_time']:.2f}s")
        print(f"  Per Sample:    {timing['per_sample_time']*1000:.2f}ms")
        print(f"  Throughput:    {timing['throughput']:.1f} samples/s")
        
        print("=" * 60)
    
    def compare_methods(self, all_results: Dict[str, Dict]) -> Dict:
        """
        Compare all evaluated methods.
        
        Args:
            all_results: Dictionary of {method_name: results}
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'methods': list(all_results.keys()),
            'classification': {},
            'uncertainty': {},
            'timing': {}
        }
        
        for method_name, results in all_results.items():
            # Classification
            comparison['classification'][method_name] = {
                'accuracy': results['classification']['accuracy'],
                'f1_score': results['classification']['f1_score'],
                'precision': results['classification']['precision'],
                'recall': results['classification']['recall']
            }
            
            if 'uncertainty' in results:
                unc = results['uncertainty']
                comparison['uncertainty'][method_name] = {
                    'coverage': unc['coverage'],
                    'avg_set_size': unc['avg_set_size'],
                    'coverage_violation': abs(unc['coverage'] - results['target_coverage']),
                    'singleton_rate': unc['efficiency']['singleton_rate']
                }

            comparison['timing'][method_name] = {
                'total_time': results['timing']['total_inference_time'],
                'throughput': results['timing']['throughput']
            }

        comparison['best'] = self._find_best_methods(all_results)
        
        return comparison
    
    def _find_best_methods(self, all_results: Dict[str, Dict]) -> Dict:
        """Find best performing methods for each metric."""
        best = {}
        
        best_acc_method = max(all_results.items(),
                             key=lambda x: x[1]['classification']['accuracy'])
        best['accuracy'] = {
            'method': best_acc_method[0],
            'value': best_acc_method[1]['classification']['accuracy']
        }

        best_f1_method = max(all_results.items(),
                            key=lambda x: x[1]['classification']['f1_score'])
        best['f1_score'] = {
            'method': best_f1_method[0],
            'value': best_f1_method[1]['classification']['f1_score']
        }

        target_coverage = 1 - self.config['conformal']['alpha']
        conformal_methods = {k: v for k, v in all_results.items() 
                           if 'uncertainty' in v}
        
        if conformal_methods:
            best_cov_method = min(conformal_methods.items(),
                                 key=lambda x: abs(x[1]['uncertainty']['coverage'] - target_coverage))
            best['coverage'] = {
                'method': best_cov_method[0],
                'value': best_cov_method[1]['uncertainty']['coverage'],
                'violation': abs(best_cov_method[1]['uncertainty']['coverage'] - target_coverage)
            }

            best_size_method = min(conformal_methods.items(),
                                  key=lambda x: x[1]['uncertainty']['avg_set_size'])
            best['set_size'] = {
                'method': best_size_method[0],
                'value': best_size_method[1]['uncertainty']['avg_set_size']
            }
        
        return best
    
    def print_comparison(self, comparison: Dict):
        """Print comparison of all methods."""
        print("\n" + "=" * 80)
        print("METHODS COMPARISON")
        print("=" * 80)
        
        print("\n1. Classification Performance:")
        print("-" * 80)
        print(f"{'Method':<25} {'Accuracy':>12} {'F1-Score':>12} {'Precision':>12} {'Recall':>12}")
        print("-" * 80)
        for method in comparison['methods']:
            clf = comparison['classification'][method]
            print(f"{method:<25} {clf['accuracy']:>12.4f} {clf['f1_score']:>12.4f} "
                  f"{clf['precision']:>12.4f} {clf['recall']:>12.4f}")
        
        if comparison['uncertainty']:
            print("\n2. Uncertainty Quantification:")
            print("-" * 80)
            print(f"{'Method':<25} {'Coverage':>12} {'Target':>12} {'Violation':>12} {'Avg Size':>12} {'Singleton':>12}")
            print("-" * 80)
            target = 1 - self.config['conformal']['alpha']
            for method in comparison['methods']:
                if method in comparison['uncertainty']:
                    unc = comparison['uncertainty'][method]
                    print(f"{method:<25} {unc['coverage']:>12.4f} {target:>12.4f} "
                          f"{unc['coverage_violation']:>12.4f} {unc['avg_set_size']:>12.4f} "
                          f"{unc['singleton_rate']:>12.4f}")
        
        print("\n3. Computational Efficiency:")
        print("-" * 80)
        print(f"{'Method':<25} {'Time (s)':>15} {'Throughput':>20}")
        print("-" * 80)
        for method in comparison['methods']:
            timing = comparison['timing'][method]
            print(f"{method:<25} {timing['total_time']:>15.2f} {timing['throughput']:>18.1f} img/s")

        print("\n4. Best Performing Methods:")
        print("-" * 80)
        for metric, info in comparison['best'].items():
            print(f"  {metric.replace('_', ' ').title()}: {info['method']} ({info['value']:.4f})")
        
        print("\n" + "=" * 80 + "\n")
