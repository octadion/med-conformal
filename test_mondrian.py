"""
Verification Test for ClassMondrianCP

Tests:
1. ClassMondrianCP creates K per-class quantiles (not 3 entropy strata)
2. StratifiedCP creates n_strata entropy-based quantiles (backward compat)
3. Fallback logic works for small classes
4. API compatibility (both work with same runner pattern)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append('.')


def create_synthetic_data(n_samples=1000, n_classes=9, seed=42):
    """Create synthetic classification data for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes
            self.fc = nn.Linear(32, num_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel(n_classes)
    
    # Create synthetic features and labels
    X = torch.randn(n_samples, 32)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Create dataloaders (split: 30% tune, 40% calib, 30% val)
    n_tune = int(n_samples * 0.3)
    n_calib = int(n_samples * 0.4)
    
    tune_ds = TensorDataset(X[:n_tune], labels[:n_tune])
    calib_ds = TensorDataset(X[n_tune:n_tune+n_calib], labels[n_tune:n_tune+n_calib])
    val_ds = TensorDataset(X[n_tune+n_calib:], labels[n_tune+n_calib:])
    
    tune_loader = DataLoader(tune_ds, batch_size=64, shuffle=False)
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # Also create a test set
    X_test = torch.randn(200, 32)
    labels_test = torch.randint(0, n_classes, (200,))
    test_ds = TensorDataset(X_test, labels_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    return model, tune_loader, calib_loader, val_loader, test_loader


def test_class_mondrian():
    """Test ClassMondrianCP has K per-class quantiles."""
    from models.conformal_wrapper import ClassMondrianCP, predict_with_sets
    
    print("=" * 60)
    print("TEST 1: ClassMondrianCP — per-class quantiles")
    print("=" * 60)
    
    n_classes = 9
    model, tune_loader, calib_loader, val_loader, test_loader = \
        create_synthetic_data(n_samples=1000, n_classes=n_classes)
    
    class_mondrian = ClassMondrianCP(
        model,
        calib_loader=calib_loader,
        alpha=0.1,
        randomized=True,
        min_class_size=5,
        device='cpu',
        tune_loader=tune_loader,
        val_loader=val_loader,
    )
    
    # Check: should have K quantiles
    assert len(class_mondrian.class_quantiles) == n_classes, \
        f"Expected {n_classes} class quantiles, got {len(class_mondrian.class_quantiles)}"
    print(f"✅ Has {n_classes} per-class quantiles (correct!)")
    
    # Check: each class should have a calibration size
    total_calib = sum(class_mondrian.class_sizes.values())
    print(f"   Total calibration samples distributed: {total_calib}")
    for k in range(n_classes):
        print(f"   Class {k}: n={class_mondrian.class_sizes[k]}, "
              f"q_hat={class_mondrian.class_quantiles[k]:.4f}")
    
    # Check: forward pass works
    _, _, probs, sets, labels = predict_with_sets(class_mondrian, test_loader)
    coverage = np.mean([labels[i] in sets[i] for i in range(len(labels))])
    avg_size = np.mean([len(s) for s in sets])
    print(f"\n   Test coverage: {coverage:.4f}")
    print(f"   Test avg size: {avg_size:.2f}")
    print(f"✅ Forward pass works!\n")
    
    return True


def test_stratified_cp():
    """Test StratifiedCP has n_strata entropy-based quantiles."""
    from models.conformal_wrapper import StratifiedCP, predict_with_sets
    
    print("=" * 60)
    print("TEST 2: StratifiedCP — entropy-based strata (renamed MondrianCP)")
    print("=" * 60)
    
    n_classes = 9
    n_strata = 3
    model, tune_loader, calib_loader, val_loader, test_loader = \
        create_synthetic_data(n_samples=1000, n_classes=n_classes)
    
    stratified_cp = StratifiedCP(
        model, tune_loader, calib_loader, val_loader,
        alpha=0.1, n_strata=n_strata, device='cpu'
    )
    
    # Check: should have n_strata quantiles (NOT n_classes)
    assert len(stratified_cp.stratum_quantiles) == n_strata, \
        f"Expected {n_strata} strata quantiles, got {len(stratified_cp.stratum_quantiles)}"
    print(f"✅ Has {n_strata} entropy strata quantiles (correct!)")
    
    # Check: entropy boundaries exist
    assert len(stratified_cp.entropy_boundaries) == n_strata - 1, \
        f"Expected {n_strata-1} boundaries, got {len(stratified_cp.entropy_boundaries)}"
    print(f"   Entropy boundaries: {stratified_cp.entropy_boundaries}")
    
    for s in range(n_strata):
        print(f"   Stratum {s}: q_hat={stratified_cp.stratum_quantiles[s]:.4f}")
    
    # Check: forward pass works
    _, _, probs, sets, labels = predict_with_sets(stratified_cp, test_loader)
    coverage = np.mean([labels[i] in sets[i] for i in range(len(labels))])
    avg_size = np.mean([len(s) for s in sets])
    print(f"\n   Test coverage: {coverage:.4f}")
    print(f"   Test avg size: {avg_size:.2f}")
    print(f"✅ Forward pass works!\n")
    
    return True


def test_backward_compat():
    """Test MondrianCP alias still works."""
    from models.conformal_wrapper import MondrianCP, StratifiedCP
    
    print("=" * 60)
    print("TEST 3: Backward compatibility (MondrianCP alias)")
    print("=" * 60)
    
    assert MondrianCP is StratifiedCP, \
        "MondrianCP should be an alias for StratifiedCP"
    print(f"✅ MondrianCP is StratifiedCP (backward compat OK)\n")
    
    return True


def test_fallback_logic():
    """Test ClassMondrianCP fallback for small classes."""
    from models.conformal_wrapper import ClassMondrianCP
    
    print("=" * 60)
    print("TEST 4: ClassMondrianCP fallback for rare classes")
    print("=" * 60)
    
    # Create data with very few samples → some classes will be rare in predictions
    n_classes = 9
    model, tune_loader, calib_loader, val_loader, _ = \
        create_synthetic_data(n_samples=100, n_classes=n_classes, seed=99)
    
    class_mondrian = ClassMondrianCP(
        model,
        calib_loader=calib_loader,
        alpha=0.1,
        randomized=True,
        min_class_size=10,  # High threshold to trigger fallbacks
        device='cpu',
    )
    
    n_fallback = len(class_mondrian.calibration_info['fallback_classes'])
    print(f"   Classes using fallback: {n_fallback}/{n_classes}")
    print(f"   Fallback classes: {class_mondrian.calibration_info['fallback_classes']}")
    
    # Verify fallback classes use global quantile
    for k in class_mondrian.calibration_info['fallback_classes']:
        assert class_mondrian.class_quantiles[k] == class_mondrian.global_q_hat, \
            f"Fallback class {k} should use global q_hat"
    
    print(f"✅ Fallback logic works correctly!\n")
    return True


def test_key_difference():
    """
    KEY TEST: Verify ClassMondrianCP and StratifiedCP produce 
    DIFFERENT numbers of groups for the same data.
    """
    from models.conformal_wrapper import ClassMondrianCP, StratifiedCP
    
    print("=" * 60)
    print("TEST 5: ClassMondrianCP vs StratifiedCP — different grouping")
    print("=" * 60)
    
    n_classes = 11  # Like OrganAMNIST
    n_strata = 3
    model, tune_loader, calib_loader, val_loader, _ = \
        create_synthetic_data(n_samples=1000, n_classes=n_classes)
    
    class_mondrian = ClassMondrianCP(
        model, calib_loader=calib_loader, alpha=0.1, device='cpu'
    )
    
    stratified_cp = StratifiedCP(
        model, tune_loader, calib_loader, val_loader,
        alpha=0.1, n_strata=n_strata, device='cpu'
    )
    
    n_class_groups = len(class_mondrian.class_quantiles)
    n_entropy_groups = len(stratified_cp.stratum_quantiles)
    
    print(f"   ClassMondrianCP groups: {n_class_groups} (= num_classes)")
    print(f"   StratifiedCP groups:    {n_entropy_groups} (= n_strata)")
    
    assert n_class_groups == n_classes, \
        f"ClassMondrianCP should have {n_classes} groups, got {n_class_groups}"
    assert n_entropy_groups == n_strata, \
        f"StratifiedCP should have {n_strata} groups, got {n_entropy_groups}"
    assert n_class_groups != n_entropy_groups, \
        "The two methods should produce different numbers of groups!"
    
    print(f"✅ Different grouping confirmed! ({n_class_groups} vs {n_entropy_groups})\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING ClassMondrianCP VERIFICATION TESTS")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("ClassMondrianCP per-class", test_class_mondrian()))
    results.append(("StratifiedCP entropy-based", test_stratified_cp()))
    results.append(("Backward compatibility", test_backward_compat()))
    results.append(("Fallback logic", test_fallback_logic()))
    results.append(("Grouping difference", test_key_difference()))
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print(f"\n{'✅ ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED!'}")
    print("=" * 60)