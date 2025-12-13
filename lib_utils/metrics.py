"""
Comprehensive metrics for classification and uncertainty quantification.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: Optional[np.ndarray] = None,
                                   num_classes: int = 11) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (n_samples, n_classes)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = precision_w
    metrics['recall_weighted'] = recall_w
    metrics['f1_weighted'] = f1_w
    
    if y_prob is not None:
        try:
            # One-vs-Rest AUC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            
            if y_true_bin.shape[1] < num_classes:
                # Pad with zeros
                pad_width = num_classes - y_true_bin.shape[1]
                y_true_bin = np.pad(y_true_bin, ((0, 0), (0, pad_width)))
            
            if y_prob.shape[1] < num_classes:
                pad_width = num_classes - y_prob.shape[1]
                y_prob = np.pad(y_prob, ((0, 0), (0, pad_width)))
            
            metrics['auc_macro'] = roc_auc_score(
                y_true_bin, y_prob, average='macro', multi_class='ovr'
            )
            metrics['auc_weighted'] = roc_auc_score(
                y_true_bin, y_prob, average='weighted', multi_class='ovr'
            )
        except Exception as e:
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
    
    return metrics


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary of per-class metrics
    """
    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class = {}
    for i, class_name in enumerate(class_names[:len(precision)]):
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    return per_class


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_coverage(prediction_sets: List[np.ndarray], y_true: np.ndarray) -> float:
    """
    Compute coverage: fraction of samples where true label is in prediction set.
    
    Args:
        prediction_sets: List of prediction sets (each is array of label indices)
        y_true: Ground truth labels
        
    Returns:
        Coverage (fraction)
    """
    covered = 0
    for pred_set, true_label in zip(prediction_sets, y_true):
        if true_label in pred_set:
            covered += 1
    
    return covered / len(y_true)


def compute_average_set_size(prediction_sets: List[np.ndarray]) -> float:
    """
    Compute average prediction set size.
    
    Args:
        prediction_sets: List of prediction sets
        
    Returns:
        Average set size
    """
    return np.mean([len(s) for s in prediction_sets])


def compute_set_size_distribution(prediction_sets: List[np.ndarray]) -> Dict[int, int]:
    """
    Compute distribution of set sizes.
    
    Args:
        prediction_sets: List of prediction sets
        
    Returns:
        Dictionary {set_size: count}
    """
    size_counts = {}
    for pred_set in prediction_sets:
        size = len(pred_set)
        size_counts[size] = size_counts.get(size, 0) + 1
    
    return dict(sorted(size_counts.items()))


def compute_conditional_coverage(prediction_sets: List[np.ndarray], 
                                 y_true: np.ndarray,
                                 num_classes: int = 11) -> Dict[int, float]:
    """
    Compute coverage per class (conditional coverage).
    
    Args:
        prediction_sets: List of prediction sets
        y_true: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Dictionary {class_id: coverage}
    """
    class_coverage = {}
    
    for class_id in range(num_classes):
        class_mask = (y_true == class_id)
        if class_mask.sum() == 0:
            continue

        covered = sum(
            y_true[i] in prediction_sets[i]
            for i in range(len(y_true))
            if class_mask[i]
        )
        
        class_coverage[class_id] = covered / class_mask.sum()
    
    return class_coverage


def compute_stratified_coverage(prediction_sets: List[np.ndarray],
                                y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute coverage stratified by set size.
    
    Args:
        prediction_sets: List of prediction sets
        y_true: Ground truth labels
        
    Returns:
        Dictionary {stratum: coverage}
    """
    strata = {
        'size_1': [],
        'size_2': [],
        'size_3': [],
        'size_4+': []
    }
    
    for i, pred_set in enumerate(prediction_sets):
        size = len(pred_set)
        covered = int(y_true[i] in pred_set)
        
        if size == 1:
            strata['size_1'].append(covered)
        elif size == 2:
            strata['size_2'].append(covered)
        elif size == 3:
            strata['size_3'].append(covered)
        else:
            strata['size_4+'].append(covered)
    
    stratified_coverage = {}
    for stratum, covered_list in strata.items():
        if len(covered_list) > 0:
            stratified_coverage[stratum] = np.mean(covered_list)
        else:
            stratified_coverage[stratum] = np.nan
    
    return stratified_coverage


def compute_efficiency(prediction_sets: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute efficiency metrics for prediction sets.
    
    Args:
        prediction_sets: List of prediction sets
        
    Returns:
        Dictionary of efficiency metrics
    """
    sizes = np.array([len(s) for s in prediction_sets])
    
    return {
        'avg_size': float(np.mean(sizes)),
        'median_size': float(np.median(sizes)),
        'std_size': float(np.std(sizes)),
        'min_size': int(np.min(sizes)),
        'max_size': int(np.max(sizes)),
        'singleton_rate': float(np.mean(sizes == 1)),
        'empty_rate': float(np.mean(sizes == 0))
    }


def compute_calibration_error(y_prob: np.ndarray, y_true: np.ndarray, 
                              n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_prob: Prediction probabilities (n_samples, n_classes)
        y_true: Ground truth labels
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_comprehensive_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  y_prob: Optional[np.ndarray],
                                  prediction_sets: Optional[List[np.ndarray]],
                                  class_names: Optional[List[str]] = None,
                                  num_classes: int = 11) -> Dict[str, any]:
    """
    Compute all metrics in one go.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        prediction_sets: Prediction sets (for conformal methods)
        class_names: List of class names
        num_classes: Number of classes
        
    Returns:
        Dictionary containing all metrics
    """
    results = {}
    
    results['classification'] = compute_classification_metrics(
        y_true, y_pred, y_prob, num_classes
    )
    
    results['per_class'] = compute_per_class_metrics(
        y_true, y_pred, class_names
    )

    results['confusion_matrix'] = compute_confusion_matrix(y_true, y_pred)
    
    if prediction_sets is not None:
        results['uncertainty'] = {
            'coverage': compute_coverage(prediction_sets, y_true),
            'avg_set_size': compute_average_set_size(prediction_sets),
            'set_size_distribution': compute_set_size_distribution(prediction_sets),
            'conditional_coverage': compute_conditional_coverage(prediction_sets, y_true, num_classes),
            'stratified_coverage': compute_stratified_coverage(prediction_sets, y_true),
            'efficiency': compute_efficiency(prediction_sets)
        }
    
    if y_prob is not None:
        results['calibration'] = {
            'ece': compute_calibration_error(y_prob, y_true)
        }
    
    return results
