"""
Challenging Dataset Experiment

Tests ES-RAPS on a more challenging / larger-K dataset.

Strategy:
  1. Try OrganMNIST3D (3D volumetric, 11 classes) — if available
  2. Fallback: ChestMNIST (14 binary labels → multi-label, but we treat as 
     multi-class for conformal prediction by using the provided label mapping)
  3. Fallback: BloodMNIST (8 classes, different modality)

Minimal eval: LAC, RAPS-Std, EntropyRAPS, ClassMondrianCP
1 seed sufficient.

Usage:
    python experiments/challenging_dataset.py
    python experiments/challenging_dataset.py --dataset chestmnist
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random
import sys
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from scipy.stats import entropy

sys.path.append('.')

from models.conformal_wrapper import (
    StandardLAC, SizeOptimizedRAPS, EntropyStratifiedRAPS,
    ClassMondrianCP, predict_with_sets
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_available_datasets():
    """Check which MedMNIST datasets are available."""
    import medmnist
    from medmnist import INFO
    
    candidates = [
        ('organmnist3d', '3D volumetric CT, 11 classes'),
        ('nodulemnist3d', '3D lung nodule, 2 classes'),
        ('synapsemnist3d', '3D synapse, 2 classes'),
        ('chestmnist', 'Chest X-ray, 14 binary labels'),
        ('bloodmnist', 'Blood cell microscopy, 8 classes'),
        ('tissuemnist', 'Kidney cortex microscopy, 8 classes'),
        ('octmnist', 'Retinal OCT, 4 classes'),
        ('retinamnist', 'Fundus camera, 5 classes'),
    ]
    
    available = []
    print("\nChecking available MedMNIST datasets:")
    for name, desc in candidates:
        if name in INFO:
            info = INFO[name]
            n_classes = len(info['label'])
            task = info['task']
            print(f"  ✅ {name}: {desc} (K={n_classes}, task={task})")
            available.append((name, n_classes, task))
        else:
            print(f"  ❌ {name}: Not found in INFO")
    
    return available


def select_dataset(preferred=None):
    """
    Select the best challenging dataset.
    
    Priority:
    1. User-specified dataset
    2. OrganMNIST3D (3D, 11 classes) — but needs 3D model, skip for now
    3. BloodMNIST (8 classes, different modality from Path/Organ)
    4. TissueMNIST (8 classes, expected ~68% accuracy → good for coverage test)
    5. ChestMNIST (14 labels — but multi-label, needs special handling)
    """
    import medmnist
    from medmnist import INFO
    
    if preferred and preferred in INFO:
        info = INFO[preferred]
        n_classes = len(info['label'])
        task = info['task']
        print(f"\nUsing requested dataset: {preferred} (K={n_classes}, task={task})")
        return preferred, n_classes, task
    
    # Priority order for 2D multi-class datasets
    priority = ['bloodmnist', 'tissuemnist', 'retinamnist', 'octmnist']
    
    for name in priority:
        if name in INFO:
            info = INFO[name]
            n_classes = len(info['label'])
            task = info['task']
            if task == 'multi-class':
                print(f"\nAuto-selected: {name} (K={n_classes}, task={task})")
                return name, n_classes, task
    
    # Last resort: chestmnist (multi-label, needs adaptation)
    if 'chestmnist' in INFO:
        print("\nFallback to ChestMNIST (multi-label → will use argmax of predictions)")
        return 'chestmnist', 14, 'multi-label, binary-class'
    
    raise RuntimeError("No suitable MedMNIST dataset found!")


class MedMNISTClassifier(nn.Module):
    """ResNet18 classifier for any MedMNIST dataset."""
    
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Handle grayscale input
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms(n_channels):
    """Get train/test transforms handling grayscale vs RGB."""
    if n_channels == 1:
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return train_transform, test_transform


def train_model(model, train_loader, val_loader, device, epochs=25, lr=0.001):
    """Train model and return best version."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1).long().to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.view(-1).long().to(device)
                preds = model(x).argmax(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Val Acc = {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"   Best Val Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, best_acc


def compute_hard_coverage(probs, sets, labels, percentile=0.90):
    """Compute coverage on hard cases (top entropy percentile)."""
    ent = entropy(probs, axis=1)
    threshold = np.quantile(ent, percentile)
    hard_mask = ent > threshold
    if hard_mask.sum() == 0:
        return float('nan')
    return np.mean([labels[i] in sets[i] for i in range(len(labels)) if hard_mask[i]])


def main():
    parser = argparse.ArgumentParser(description='Challenging dataset experiment')
    parser.add_argument('--dataset', type=str, default=None,
                        help='MedMNIST dataset name (auto-select if not specified)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    
    if args.save_dir is None:
        drive_path = Path('/content/drive/MyDrive/MedConformal_Final_Submission5')
        if drive_path.exists():
            save_dir = drive_path / 'challenging_dataset'
        else:
            save_dir = Path('./challenging_dataset_results')
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CHALLENGING DATASET EXPERIMENT")
    print("=" * 70)
    
    # Check available datasets
    available = check_available_datasets()
    
    # Select dataset
    dataset_name, n_classes, task = select_dataset(args.dataset)
    
    # Load data
    print(f"\n[1/4] Loading {dataset_name}...")
    import medmnist
    from medmnist import INFO
    
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']
    
    train_transform, test_transform = get_transforms(n_channels)
    
    train_dataset = DataClass(split='train', transform=train_transform,
                              download=True, root='./data')
    val_dataset = DataClass(split='val', transform=test_transform,
                            download=True, root='./data')
    test_dataset = DataClass(split='test', transform=test_transform,
                             download=True, root='./data')
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"   Classes: {n_classes}, Channels: {n_channels}, Task: {task}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader_full = DataLoader(val_dataset, batch_size=128, shuffle=False,
                                 num_workers=4)
    test_loader_full = DataLoader(test_dataset, batch_size=128, shuffle=False,
                                  num_workers=4)
    
    # Train or load model
    print(f"\n[2/4] Training model...")
    model = MedMNISTClassifier(n_classes, in_channels=3).to(device)  # Always 3ch after transform
    
    model_path = save_dir / f'{dataset_name}_model.pth'
    if model_path.exists():
        print(f"   Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Compute accuracy
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader_full:
                x, y = x.to(device), y.view(-1).long().to(device)
                preds = model(x).argmax(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        print(f"   Test Accuracy: {accuracy:.4f}")
    else:
        model, accuracy = train_model(model, train_loader, val_loader_full, 
                                       device, epochs=args.epochs)
        torch.save(model.state_dict(), model_path)
        print(f"   Model saved to {model_path}")
        
        # Recompute on test
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader_full:
                x, y = x.to(device), y.view(-1).long().to(device)
                preds = model(x).argmax(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        print(f"   Test Accuracy: {accuracy:.4f}")
    
    # 3-way split from val+test combined for conformal
    # Use val set for conformal splits, test set for final eval
    print(f"\n[3/4] Setting up conformal splits...")
    
    n_val = len(val_dataset)
    indices = np.random.permutation(n_val)
    
    n_tune = int(n_val * 0.3)
    n_calib = int(n_val * 0.4)
    
    tune_ds = Subset(val_dataset, indices[:n_tune])
    calib_ds = Subset(val_dataset, indices[n_tune:n_tune + n_calib])
    val_split_ds = Subset(val_dataset, indices[n_tune + n_calib:])
    
    tune_loader = DataLoader(tune_ds, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
    val_split_loader = DataLoader(val_split_ds, batch_size=32, shuffle=False)
    test_loader = test_loader_full
    
    print(f"   Tune: {len(tune_ds)}, Calib: {len(calib_ds)}, "
          f"Val: {len(val_split_ds)}, Test: {len(test_dataset)}")
    
    # Run conformal methods
    print(f"\n[4/4] Running conformal methods (α={args.alpha})...")
    
    set_seed(args.seed)
    
    methods = {}
    
    print("\n   LAC...")
    methods['LAC'] = StandardLAC(model, calib_loader, alpha=args.alpha, device=device)
    
    print("\n   RAPS-Standard...")
    methods['RAPS-Std'] = SizeOptimizedRAPS(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=args.alpha, k_reg=2, randomized=True, device=device
    )
    
    print("\n   EntropyRAPS (Ours)...")
    methods['EntropyRAPS'] = EntropyStratifiedRAPS(
        model, tune_loader, calib_loader, val_split_loader,
        alpha=args.alpha, k_reg=2, randomized=True, device=device
    )
    
    print("\n   ClassMondrianCP...")
    methods['ClassMondrian'] = ClassMondrianCP(
        model, calib_loader=calib_loader, alpha=args.alpha,
        randomized=True, min_class_size=5, device=device
    )
    
    # Evaluate
    results = []
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_name.upper()} (K={n_classes}, Accuracy={accuracy:.4f})")
    print(f"{'='*70}")
    print(f"\n{'Method':<18} {'Coverage':<12} {'Hard Cov':<12} {'Avg Size':<10}")
    print("-" * 55)
    
    for method_name, conf_model in methods.items():
        _, _, probs, sets, labels = predict_with_sets(conf_model, test_loader)
        
        cov = np.mean([labels[i] in sets[i] for i in range(len(labels))])
        avg_size = np.mean([len(s) for s in sets])
        hard_cov = compute_hard_coverage(probs, sets, labels)
        
        print(f"{method_name:<18} {cov:<12.4f} {hard_cov:<12.4f} {avg_size:<10.2f}")
        
        results.append({
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'K': n_classes,
            'Method': method_name,
            'Coverage': cov,
            'Hard_Coverage': hard_cov,
            'Avg_Size': avg_size,
        })
        
        # Extra: get lambda for RAPS methods
        if hasattr(conf_model, 'lamda_star'):
            results[-1]['lambda_star'] = conf_model.lamda_star
    
    print("-" * 55)
    
    # Save
    df = pd.DataFrame(results)
    csv_path = save_dir / f'challenging_{dataset_name}_results.csv'
    df.to_csv(csv_path, index=False)
    
    # LaTeX table
    print(f"\n\n% LaTeX table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Results on {dataset_name} ($K={n_classes}$, "
          f"Accuracy={accuracy:.1%}, $\\alpha={args.alpha}$)}}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Method & Coverage & Hard Cov. & Avg. Size \\\\")
    print("\\midrule")
    for _, row in df.iterrows():
        print(f"{row['Method']} & {row['Coverage']:.4f} & "
              f"{row['Hard_Coverage']:.4f} & {row['Avg_Size']:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print(f"\n✅ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()