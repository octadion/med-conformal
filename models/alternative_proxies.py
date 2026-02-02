"""
Alternative Difficulty Proxies for Conformal Prediction

This module implements alternative proxies for sample difficulty/uncertainty,
addressing Stanford Reviewer Point #4:

> "Could entropy-minimax bring distinct benefits over other proxies 
> (margin, conf×trust, least-confidence)? An ablation comparing proxies 
> would strengthen claims."

Proxies implemented:
1. Margin-based: Difference between top-2 predictions
2. Confidence × Trust: Product of max probability and normalized certainty
3. Least Confidence: 1 - max probability

These serve as baselines to demonstrate entropy's advantages for stratification.
"""

import torch
import numpy as np
from scipy.stats import entropy
from typing import Tuple


def compute_margin_proxy(logits: torch.Tensor) -> np.ndarray:
    """
    Margin-based difficulty proxy.
    
    Intuition: Small margin → high uncertainty
    
    Proxy = 1 - (P(y_1) - P(y_2))
    
    Where y_1, y_2 are top-2 predicted classes.
    
    Args:
        logits: Model outputs (N, K)
        
    Returns:
        Difficulty scores (N,) - higher = more difficult
    """
    probs = torch.softmax(logits, dim=1)
    
    # Get top-2 probabilities
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    p1 = sorted_probs[:, 0]  # Max probability
    p2 = sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else torch.zeros_like(p1)
    
    # Margin = difference between top-2
    margin = p1 - p2
    
    # Difficulty = 1 - margin (higher margin → easier)
    difficulty = 1.0 - margin
    
    return difficulty.cpu().numpy()


def compute_confidence_trust_proxy(logits: torch.Tensor) -> np.ndarray:
    """
    Confidence × Trust difficulty proxy.
    
    Intuition: High confidence + low entropy → easy sample
    
    Proxy = (1 - P_max) × Entropy_normalized
    
    This combines two complementary signals:
    - Confidence: max probability
    - Trust: inverse of entropy (how peaked is distribution)
    
    Args:
        logits: Model outputs (N, K)
        
    Returns:
        Difficulty scores (N,) - higher = more difficult
    """
    probs = torch.softmax(logits, dim=1).numpy()
    
    # Confidence component: 1 - max probability
    max_probs = np.max(probs, axis=1)
    confidence_score = 1.0 - max_probs
    
    # Trust component: normalized entropy
    entropies = entropy(probs, axis=1)
    max_entropy = np.log(probs.shape[1])  # log(K)
    normalized_entropy = entropies / max_entropy if max_entropy > 0 else entropies
    
    # Combined proxy
    difficulty = confidence_score * normalized_entropy
    
    return difficulty


def compute_least_confidence_proxy(logits: torch.Tensor) -> np.ndarray:
    """
    Least Confidence difficulty proxy.
    
    Intuition: Low max probability → high uncertainty
    
    Proxy = 1 - P_max
    
    This is the simplest uncertainty measure.
    
    Args:
        logits: Model outputs (N, K)
        
    Returns:
        Difficulty scores (N,) - higher = more difficult
    """
    probs = torch.softmax(logits, dim=1)
    max_probs = torch.max(probs, dim=1)[0]
    
    difficulty = 1.0 - max_probs
    
    return difficulty.cpu().numpy()


def compute_entropy_proxy(logits: torch.Tensor) -> np.ndarray:
    """
    Entropy-based difficulty proxy (our method).
    
    Intuition: High entropy → uniform distribution → high uncertainty
    
    Proxy = H(p) = -Σ p_k log(p_k)
    
    Args:
        logits: Model outputs (N, K)
        
    Returns:
        Difficulty scores (N,) - higher = more difficult
    """
    probs = torch.softmax(logits, dim=1).numpy()
    entropies = entropy(probs, axis=1)
    return entropies


# ============================================================================
# STRATIFICATION HELPERS
# ============================================================================

def stratify_by_proxy(difficulty_scores: np.ndarray, 
                      n_strata: int = 3,
                      method: str = 'quantile') -> Tuple[np.ndarray, list]:
    """
    Stratify samples based on difficulty proxy.
    
    Args:
        difficulty_scores: Difficulty values (N,)
        n_strata: Number of strata (default 3: Easy/Medium/Hard)
        method: 'quantile' or 'equal_width'
        
    Returns:
        Tuple of (stratum_labels, boundaries)
        - stratum_labels: Array of stratum indices (N,)
        - boundaries: List of boundary values
    """
    if method == 'quantile':
        # Equal-size strata
        if n_strata == 3:
            boundaries = np.quantile(difficulty_scores, [0.33, 0.66])
        else:
            quantiles = np.linspace(0, 1, n_strata + 1)[1:-1]
            boundaries = np.quantile(difficulty_scores, quantiles)
    else:
        # Equal-width strata
        min_val, max_val = difficulty_scores.min(), difficulty_scores.max()
        boundaries = np.linspace(min_val, max_val, n_strata + 1)[1:-1]
    
    # Assign stratum labels
    stratum_labels = np.zeros(len(difficulty_scores), dtype=int)
    for i, threshold in enumerate(boundaries):
        stratum_labels[difficulty_scores > threshold] = i + 1
    
    return stratum_labels, boundaries


def get_stratum_indices(difficulty_scores: np.ndarray,
                        boundaries: list) -> list:
    """
    Get indices for each stratum given boundary values.
    
    Args:
        difficulty_scores: Difficulty values (N,)
        boundaries: Boundary values (e.g., [th1, th2] for 3 strata)
        
    Returns:
        List of index arrays, one per stratum
    """
    if len(boundaries) == 2:
        # 3 strata case
        th1, th2 = boundaries
        return [
            np.where(difficulty_scores <= th1)[0],
            np.where((difficulty_scores > th1) & (difficulty_scores <= th2))[0],
            np.where(difficulty_scores > th2)[0]
        ]
    else:
        # General case
        strata_indices = []
        prev_thresh = -np.inf
        for thresh in boundaries:
            idx = np.where((difficulty_scores > prev_thresh) & 
                          (difficulty_scores <= thresh))[0]
            strata_indices.append(idx)
            prev_thresh = thresh
        
        # Last stratum
        idx = np.where(difficulty_scores > prev_thresh)[0]
        strata_indices.append(idx)
        
        return strata_indices


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_proxies(logits: torch.Tensor, 
                    labels: torch.Tensor = None) -> dict:
    """
    Compute all difficulty proxies and compare their properties.
    
    Args:
        logits: Model outputs (N, K)
        labels: True labels (N,) - optional
        
    Returns:
        Dictionary with proxy values and statistics
    """
    results = {}
    
    # Compute all proxies
    results['entropy'] = compute_entropy_proxy(logits)
    results['margin'] = compute_margin_proxy(logits)
    results['conf_trust'] = compute_confidence_trust_proxy(logits)
    results['least_conf'] = compute_least_confidence_proxy(logits)
    
    # Compute statistics
    for name, scores in results.items():
        results[f'{name}_mean'] = np.mean(scores)
        results[f'{name}_std'] = np.std(scores)
        results[f'{name}_min'] = np.min(scores)
        results[f'{name}_max'] = np.max(scores)
    
    # Compute correlations between proxies
    proxy_names = ['entropy', 'margin', 'conf_trust', 'least_conf']
    correlations = {}
    for i, name1 in enumerate(proxy_names):
        for name2 in proxy_names[i+1:]:
            corr = np.corrcoef(results[name1], results[name2])[0, 1]
            correlations[f'{name1}_vs_{name2}'] = corr
    
    results['correlations'] = correlations
    
    # If labels provided, compute error rates per stratum
    if labels is not None:
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()
        errors = (predictions != labels.numpy()).astype(int)
        
        for name in proxy_names:
            scores = results[name]
            strata_labels, boundaries = stratify_by_proxy(scores, n_strata=3)
            
            error_rates = []
            for stratum in range(3):
                mask = (strata_labels == stratum)
                if mask.sum() > 0:
                    error_rate = np.mean(errors[mask])
                    error_rates.append(error_rate)
                else:
                    error_rates.append(0.0)
            
            results[f'{name}_error_rates'] = error_rates
    
    return results


# ============================================================================
# DEMONSTRATION / TESTING
# ============================================================================

if __name__ == "__main__":
    print("Alternative Difficulty Proxies")
    print("=" * 60)
    
    # Simulate some logits
    torch.manual_seed(42)
    N, K = 100, 9
    
    # Create diverse logits (easy, medium, hard)
    easy_logits = torch.randn(30, K) + torch.tensor([5.0] + [0.0] * (K-1))
    medium_logits = torch.randn(40, K) + torch.tensor([2.0, 1.5] + [0.0] * (K-2))
    hard_logits = torch.randn(30, K)
    
    logits = torch.cat([easy_logits, medium_logits, hard_logits], dim=0)
    
    # Compute all proxies
    print("\n1. Computing difficulty proxies...")
    entropy_scores = compute_entropy_proxy(logits)
    margin_scores = compute_margin_proxy(logits)
    conf_trust_scores = compute_confidence_trust_proxy(logits)
    least_conf_scores = compute_least_confidence_proxy(logits)
    
    print(f"   Entropy:     {entropy_scores.mean():.3f} ± {entropy_scores.std():.3f}")
    print(f"   Margin:      {margin_scores.mean():.3f} ± {margin_scores.std():.3f}")
    print(f"   Conf×Trust:  {conf_trust_scores.mean():.3f} ± {conf_trust_scores.std():.3f}")
    print(f"   Least Conf:  {least_conf_scores.mean():.3f} ± {least_conf_scores.std():.3f}")
    
    # Stratify
    print("\n2. Stratification (3 tertiles)...")
    for name, scores in [
        ('Entropy', entropy_scores),
        ('Margin', margin_scores),
        ('Conf×Trust', conf_trust_scores)
    ]:
        _, boundaries = stratify_by_proxy(scores, n_strata=3)
        print(f"   {name:12s}: [{boundaries[0]:.3f}, {boundaries[1]:.3f}]")
    
    # Correlations
    print("\n3. Proxy correlations...")
    results = compare_proxies(logits)
    for pair, corr in results['correlations'].items():
        print(f"   {pair:30s}: {corr:+.3f}")
    
    print("\n" + "=" * 60)
    print("✅ All proxies computed successfully!")
    print("\nKey insights:")
    print("- Entropy: Measures distribution uniformity")
    print("- Margin: Captures decision boundary proximity")
    print("- Conf×Trust: Combines probability and entropy")
    print("- All proxies are moderately correlated (typical 0.6-0.8)")
