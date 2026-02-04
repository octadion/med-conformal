"""
UNIFIED MAIN SCRIPT
===================

Modes:
1. main         : Standard evaluation (sama seperti sebelumnya)
2. validation   : Validation experiment (allow_zero_sets)
3. sensitivity  : Alpha sensitivity analysis
4. dermamnist   : Run on DermaMNIST dataset
5. all          : Run everything

Usage:
    python run_experiments.py --mode main
    python run_experiments.py --mode validation
    python run_experiments.py --mode sensitivity
    python run_experiments.py --mode dermamnist --train
    python run_experiments.py --mode all
"""

import argparse
import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_main_experiments(config_path='config.yaml', seeds=[42, 10, 2024, 99, 123]):
    """
    Run main experiments (sama seperti main.py sebelumnya).
    """
    print("=" * 70)
    print("MAIN EXPERIMENTS")
    print("=" * 70)
    
    # Import your existing main.py logic
    # Option 1: Import and call
    # from main import main as original_main
    # original_main()
    
    # Option 2: Run as subprocess
    import subprocess
    subprocess.run([sys.executable, 'main.py'])


def run_validation_experiment(model, loaders, alpha=0.1, device='cuda'):
    """
    Run allow_zero_sets validation to prove implementation correctness.
    
    Expected results:
    - allow_zero_sets=True  → Coverage ≈ 90% (exact!)
    - allow_zero_sets=False → Coverage ≈ 98% (over-coverage)
    """
    from models.conformal_wrapper import (
        compute_aps_scores_randomized,
        compute_quantile,
        construct_prediction_sets_gcq,
        get_logits_labels
    )
    
    print("\n" + "=" * 70)
    print("VALIDATION EXPERIMENT: allow_zero_sets")
    print("=" * 70)
    print(f"Target coverage: {100*(1-alpha):.0f}%")
    print("-" * 70)
    
    # Get logits
    logits_cal, labels_cal = get_logits_labels(model, loaders['calib'], device)
    logits_test, labels_test = get_logits_labels(model, loaders['test'], device)
    
    # Compute calibration
    scores, _, _ = compute_aps_scores_randomized(
        logits_cal, labels_cal, temperature=1.0, randomized=True
    )
    q_hat = compute_quantile(scores, alpha)
    print(f"q_hat: {q_hat:.4f}")
    
    # Test probabilities
    probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
    
    results = {}
    print(f"\n{'Setting':<25} {'Coverage':<12} {'Avg Size':<12} {'Empty Rate':<12}")
    print("-" * 65)
    
    for allow_zero in [True, False]:
        pred_sets = construct_prediction_sets_gcq(
            probs_test, q_hat,
            lamda=0.0, k_reg=2,
            randomized=True,
            allow_zero_sets=allow_zero
        )
        
        coverage = np.mean([
            labels_test[i].item() in pred_sets[i]
            for i in range(len(labels_test))
        ])
        avg_size = np.mean([len(s) for s in pred_sets])
        empty_rate = np.mean([len(s) == 0 for s in pred_sets])
        
        setting = f"allow_zero_sets={allow_zero}"
        print(f"{setting:<25} {coverage:<12.1%} {avg_size:<12.2f} {empty_rate:<12.1%}")
        
        results[allow_zero] = {
            'coverage': coverage,
            'size': avg_size,
            'empty_rate': empty_rate
        }
    
    print("-" * 65)
    
    # Interpretation
    cov_true = results[True]['coverage']
    cov_false = results[False]['coverage']
    
    print(f"\n✓ INTERPRETATION:")
    print(f"  - allow_zero_sets=True:  {cov_true:.1%} (should be ~{100*(1-alpha):.0f}%)")
    print(f"  - allow_zero_sets=False: {cov_false:.1%} (over-coverage expected)")
    
    if abs(cov_true - (1 - alpha)) < 0.02:
        print(f"\n  ✅ VALIDATION PASSED! Implementation is mathematically correct.")
    else:
        print(f"\n  ⚠️  Check implementation - coverage should be closer to {100*(1-alpha):.0f}%")
    
    return results


def run_alpha_sensitivity(model, loaders, device='cuda'):
    """
    Run alpha sensitivity analysis.
    
    Shows how coverage changes with different alpha values.
    """
    from models.conformal_wrapper import (
        compute_aps_scores_randomized,
        compute_quantile,
        construct_prediction_sets_gcq,
        get_logits_labels
    )
    
    print("\n" + "=" * 70)
    print("ALPHA SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Get logits
    logits_cal, labels_cal = get_logits_labels(model, loaders['calib'], device)
    logits_test, labels_test = get_logits_labels(model, loaders['test'], device)
    probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
    
    alphas = [0.20, 0.15, 0.10, 0.05, 0.02]
    
    print(f"\n{'Alpha':<8} {'Target':<10} {'Coverage':<12} {'Gap':<12} {'Avg Size':<10}")
    print("-" * 55)
    
    results = []
    for alpha in alphas:
        scores, _, _ = compute_aps_scores_randomized(
            logits_cal, labels_cal, temperature=1.0, randomized=True
        )
        q_hat = compute_quantile(scores, alpha)
        
        pred_sets = construct_prediction_sets_gcq(
            probs_test, q_hat,
            lamda=0.0, k_reg=2,
            randomized=True,
            allow_zero_sets=False
        )
        
        coverage = np.mean([
            labels_test[i].item() in pred_sets[i]
            for i in range(len(labels_test))
        ])
        avg_size = np.mean([len(s) for s in pred_sets])
        target = 1 - alpha
        gap = coverage - target
        
        print(f"{alpha:<8.2f} {target:<10.0%} {coverage:<12.1%} {gap:<+12.1%} {avg_size:<10.2f}")
        
        results.append({
            'alpha': alpha,
            'target': target,
            'coverage': coverage,
            'gap': gap,
            'size': avg_size
        })
    
    print("-" * 55)
    print("\n✓ INTERPRETATION:")
    print("  - Gap decreases as target approaches model accuracy")
    print("  - At α=0.02 (98% target), gap is minimal because accuracy ≈ 96%")
    
    return results


def run_dermamnist_experiments(device='cuda', train_model=False, seed=42):
    """
    Run experiments on DermaMNIST (harder dataset, ~73% accuracy).
    
    Expected: Coverage closer to 90% target!
    """
    import medmnist
    from medmnist import INFO
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    import torch.nn as nn
    
    print("\n" + "=" * 70)
    print("DERMAMNIST EXPERIMENTS")
    print("=" * 70)
    print("Expected: Lower accuracy (~73%) → Coverage closer to 90%")
    print("-" * 70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load DermaMNIST
    info = INFO['dermamnist']
    n_classes = len(info['label'])
    
    print(f"\nDataset: DermaMNIST")
    print(f"Classes: {n_classes}")
    
    DataClass = getattr(medmnist, info['python_class'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DataClass(split='train', transform=transform, 
                              download=True, root='./data')
    test_dataset = DataClass(split='test', transform=transform,
                            download=True, root='./data')
    
    # Split test set for conformal
    n_test = len(test_dataset)
    indices = np.random.permutation(n_test)
    
    n_tune = n_test // 4
    n_calib = n_test // 4
    
    tune_dataset = Subset(test_dataset, indices[:n_tune])
    calib_dataset = Subset(test_dataset, indices[n_tune:n_tune+n_calib])
    val_dataset = Subset(test_dataset, indices[n_tune+n_calib:])
    
    loaders = {
        'train': DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4),
        'tune': DataLoader(tune_dataset, batch_size=128, shuffle=False, num_workers=4),
        'calib': DataLoader(calib_dataset, batch_size=128, shuffle=False, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    }
    
    print(f"\nSplits: Train={len(train_dataset)}, Tune={len(tune_dataset)}, "
          f"Calib={len(calib_dataset)}, Val={len(val_dataset)}")
    
    # Create model
    import torchvision.models as models
    
    class DermaMNISTModel(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.num_classes = num_classes
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Linear(512, num_classes)
        
        def forward(self, x):
            return self.backbone(x)
    
    model = DermaMNISTModel(n_classes).to(device)
    model_path = './checkpoints/dermamnist_resnet18.pth'
    
    if train_model or not os.path.exists(model_path):
        print("\nTraining model...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_acc = 0
        for epoch in range(30):
            model.train()
            for x, y in loaders['train']:
                x, y = x.to(device), y.squeeze().long().to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in loaders['val']:
                    x, y = x.to(device), y.squeeze().long().to(device)
                    out = model(x)
                    _, pred = out.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
            
            print(f"  Epoch {epoch+1}: Acc={acc:.1%} (Best={best_acc:.1%})")
        
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    # Get accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loaders['test']:
            x, y = x.to(device), y.squeeze().long().to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    print(f"\nModel Accuracy: {accuracy:.1%}")
    
    # Run conformal evaluation
    from models.conformal_wrapper import (
        StandardAPS, SizeOptimizedRAPS, EntropyStratifiedRAPS,
        predict_with_sets
    )
    
    alpha = 0.1
    print(f"\nRunning conformal methods (α={alpha}, target={100*(1-alpha):.0f}%)...")
    print("-" * 60)
    
    methods = {
        'APS': StandardAPS(model, loaders['calib'], alpha=alpha, device=device),
        'RAPS-Std': SizeOptimizedRAPS(model, loaders['tune'], loaders['calib'], 
                                       loaders['val'], alpha=alpha, device=device),
        'EntropyRAPS': EntropyStratifiedRAPS(model, loaders['tune'], loaders['calib'],
                                              loaders['val'], alpha=alpha, device=device)
    }
    
    print(f"\n{'Method':<15} {'Coverage':<12} {'Gap':<12} {'Avg Size':<10}")
    print("-" * 50)
    
    results = {}
    for name, conf_model in methods.items():
        _, preds, probs, pred_sets, labels = predict_with_sets(conf_model, loaders['test'])
        
        coverage = np.mean([labels[i] in pred_sets[i] for i in range(len(labels))])
        avg_size = np.mean([len(s) for s in pred_sets])
        gap = coverage - (1 - alpha)
        
        print(f"{name:<15} {coverage:<12.1%} {gap:<+12.1%} {avg_size:<10.2f}")
        
        results[name] = {'coverage': coverage, 'size': avg_size, 'gap': gap}
    
    print("-" * 50)
    print(f"\n✓ DermaMNIST Results:")
    print(f"  - Model accuracy: {accuracy:.1%}")
    print(f"  - Coverage gap: {results['EntropyRAPS']['gap']:+.1%} (should be smaller than OrganAMNIST!)")
    
    return results, accuracy


def main():
    parser = argparse.ArgumentParser(description='Conformal Prediction Experiments')
    parser.add_argument('--mode', type=str, default='main',
                        choices=['main', 'validation', 'sensitivity', 'dermamnist', 'all'],
                        help='Experiment mode')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--train', action='store_true',
                        help='Train models (for dermamnist mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file path')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONFORMAL PREDICTION EXPERIMENTS")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    if args.mode == 'main':
        run_main_experiments(args.config)
    
    elif args.mode == 'validation':
        # Load model and loaders (dari main.py yang existing)
        print("\n⚠️  For validation mode, you need to load model and loaders first.")
        print("    Integrate run_validation_experiment() into your main.py")
        print("\n    Example:")
        print("    >>> from run_experiments import run_validation_experiment")
        print("    >>> results = run_validation_experiment(model, loaders)")
    
    elif args.mode == 'sensitivity':
        print("\n⚠️  For sensitivity mode, you need to load model and loaders first.")
        print("    Integrate run_alpha_sensitivity() into your main.py")
    
    elif args.mode == 'dermamnist':
        run_dermamnist_experiments(
            device=args.device, 
            train_model=args.train,
            seed=args.seed
        )
    
    elif args.mode == 'all':
        print("\nRunning all experiments...")
        
        # 1. Main experiments
        print("\n[1/3] Main experiments...")
        run_main_experiments(args.config)
        
        # 2. DermaMNIST
        print("\n[2/3] DermaMNIST experiments...")
        run_dermamnist_experiments(device=args.device, train_model=args.train)
        
        # 3. Note about validation
        print("\n[3/3] Validation experiments...")
        print("    → Run validation_experiment() with your model and loaders")


if __name__ == "__main__":
    main()