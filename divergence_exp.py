import torch
import yaml
import sys
from pathlib import Path
sys.path.append('.')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model
from models.conformal_wrapper import SizeOptimizedRAPS, EntropyStratifiedRAPS, predict_with_sets
import numpy as np
from scipy.stats import entropy
from torch.utils.data import Subset, DataLoader


class ImbalancedDifficultyDataset:
    """Create dataset with imbalanced difficulty distribution."""
    
    def __init__(self, base_dataset, model, 
                 hard_ratio=0.1, medium_ratio=0.2, easy_ratio=0.9,
                 device='cuda', seed=42):
        
        self.base_dataset = base_dataset
        self.model = model
        self.device = device
        
        print(f"[Imbalanced Dataset] Creating difficulty imbalance...")
        print(f"   Target ratios - Easy: {easy_ratio:.0%}, Medium: {medium_ratio:.0%}, Hard: {hard_ratio:.0%}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        entropies = self._compute_entropy_for_dataset()
        
        t1, t2 = np.quantile(entropies, [0.33, 0.66])
        
        easy_mask = entropies <= t1
        medium_mask = (entropies > t1) & (entropies <= t2)
        hard_mask = entropies > t2
        
        easy_indices = np.where(easy_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        hard_indices = np.where(hard_mask)[0]
        
        print(f"   Original counts - Easy: {len(easy_indices)}, Medium: {len(medium_indices)}, Hard: {len(hard_indices)}")
        
        n_easy = int(len(easy_indices) * easy_ratio)
        n_medium = int(len(medium_indices) * medium_ratio)
        n_hard = int(len(hard_indices) * hard_ratio)
        
        selected_easy = np.random.choice(easy_indices, n_easy, replace=False)
        selected_medium = np.random.choice(medium_indices, n_medium, replace=False)
        selected_hard = np.random.choice(hard_indices, n_hard, replace=False)
        
        self.selected_indices = np.concatenate([selected_easy, selected_medium, selected_hard])
        np.random.shuffle(self.selected_indices)
        
        self.easy_mask_new = np.isin(self.selected_indices, selected_easy)
        self.medium_mask_new = np.isin(self.selected_indices, selected_medium)
        self.hard_mask_new = np.isin(self.selected_indices, selected_hard)
        
        print(f"   Imbalanced counts - Easy: {n_easy}, Medium: {n_medium}, Hard: {n_hard}")
        print(f"   New distribution - Easy: {n_easy/(n_easy+n_medium+n_hard):.1%}, "
              f"Medium: {n_medium/(n_easy+n_medium+n_hard):.1%}, "
              f"Hard: {n_hard/(n_easy+n_medium+n_hard):.1%}")
    
    def _compute_entropy_for_dataset(self):
        self.model.eval()
        entropies = []
        
        loader = DataLoader(self.base_dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_entropy = entropy(probs, axis=1)
                entropies.extend(batch_entropy)
        
        return np.array(entropies)
    
    def get_subset(self):
        return Subset(self.base_dataset, self.selected_indices)
    
    def get_stratum_info(self):
        return {
            'easy_count': self.easy_mask_new.sum(),
            'medium_count': self.medium_mask_new.sum(),
            'hard_count': self.hard_mask_new.sum(),
        }


def main():
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['dataset'] = 'pathmnist'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("[1/5] Loading data...")
    data_loader = MedMNISTDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    print("[2/5] Loading trained model...")
    model = get_model(config, device)

    # Try multiple checkpoint paths
    checkpoint_paths = [
        Path('/content/drive/MyDrive/MedConformal_Final_Submission5/pathmnist_final/models/best_model.pth'),
        Path('/content/drive/MyDrive/MedConformal_Final_Submission2/pathmnist_final/models/pathmnist_exp_20260122_153413_best_model.pth'),
        Path('./quick_test_output/pathmnist_model.pth'),
    ]
    
    model_loaded = False
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            print(f"   Loading from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("   ✅ Model loaded successfully!")
            model_loaded = True
            break
    
    if not model_loaded:
        print("   ⚠️ Checkpoint not found! Run main.py or quick_test_pathmnist.py first.")
        return
    
    trained_model = model
    trained_model.eval()
    
    # Create IMBALANCED dataset
    print("\n[3/5] Creating imbalanced difficulty dataset...")
    imb_val = ImbalancedDifficultyDataset(
        val_loader.dataset, trained_model,
        hard_ratio=0.05, medium_ratio=0.1, easy_ratio=0.95,
        device=device, seed=42
    )
    imb_test = ImbalancedDifficultyDataset(
        test_loader.dataset, trained_model,
        hard_ratio=0.05, medium_ratio=0.1, easy_ratio=0.95,
        device=device, seed=43
    )
    
    # 3-WAY SPLIT (FIXED!)
    print("\n[4/5] Creating 3-way split...")
    imb_val_ds = imb_val.get_subset()
    n = len(imb_val_ds)
    
    # Split: 30% tune, 40% calib, 30% val
    n_tune = int(n * 0.3)
    n_calib = int(n * 0.4)
    n_val = n - n_tune - n_calib
    
    indices = np.random.permutation(n)
    tune_indices = indices[:n_tune]
    calib_indices = indices[n_tune:n_tune+n_calib]
    val_indices = indices[n_tune+n_calib:]
    
    # Create subsets
    tune_ds = Subset(imb_val_ds, tune_indices)
    calib_ds = Subset(imb_val_ds, calib_indices)
    val_ds = Subset(imb_val_ds, val_indices)
    
    tune_loader = DataLoader(tune_ds, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
    val_split_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_imb_loader = DataLoader(imb_test.get_subset(), batch_size=32, shuffle=False)
    
    print(f"   Tune: {len(tune_ds)}, Calib: {len(calib_ds)}, Val: {len(val_ds)}")
    
    # Run methods (FIXED API!)
    print("\n[5/5] Running conformal methods...")
    
    raps = SizeOptimizedRAPS(
        trained_model, 
        tune_loader,      # tune_loader first
        calib_loader,     # then calib_loader
        val_split_loader, # then val_loader
        alpha=0.1, 
        k_reg=2, 
        randomized=True,
        device=device
    )
    
    entropy_raps = EntropyStratifiedRAPS(
        trained_model, 
        tune_loader,      # tune_loader first
        calib_loader,     # then calib_loader
        val_split_loader, # then val_loader
        alpha=0.1, 
        k_reg=2, 
        randomized=True,
        device=device
    )
    
    print(f"\n{'='*60}")
    print("DIVERGENCE RESULTS:")
    print(f"{'='*60}")
    print(f"Standard RAPS λ: {raps.lamda_star:.5f}")
    print(f"EntropyRAPS λ:   {entropy_raps.lamda_star:.5f}")
    print(f"Difference:      {abs(raps.lamda_star - entropy_raps.lamda_star):.5f}")
    
    if abs(raps.lamda_star - entropy_raps.lamda_star) > 0.01:
        print("\n✅ DIVERGENCE DETECTED!")
        print("   EntropyRAPS selects different λ than Standard RAPS")
    else:
        print("\n⚠️ No significant divergence (try more extreme imbalance)")
    
    # Evaluate coverage
    print(f"\n{'='*60}")
    print("COVERAGE COMPARISON:")
    print(f"{'='*60}")
    
    # RAPS
    _, _, probs_raps, sets_raps, labels = predict_with_sets(raps, test_imb_loader)
    cov_raps = np.mean([labels[i] in sets_raps[i] for i in range(len(labels))])
    size_raps = np.mean([len(sets_raps[i]) for i in range(len(labels))])
    
    # EntropyRAPS
    _, _, probs_ent, sets_ent, labels = predict_with_sets(entropy_raps, test_imb_loader)
    cov_ent = np.mean([labels[i] in sets_ent[i] for i in range(len(labels))])
    size_ent = np.mean([len(sets_ent[i]) for i in range(len(labels))])
    
    # Hard-case coverage
    ent_vals = entropy(probs_raps, axis=1)
    hard_mask = ent_vals > np.quantile(ent_vals, 0.9)  # Top 10% hard
    
    hard_cov_raps = np.mean([labels[i] in sets_raps[i] for i in range(len(labels)) if hard_mask[i]])
    hard_cov_ent = np.mean([labels[i] in sets_ent[i] for i in range(len(labels)) if hard_mask[i]])
    
    print(f"\n{'Method':<15} {'Overall Cov':<15} {'Hard Cov':<15} {'Avg Size':<10}")
    print("-" * 55)
    print(f"{'RAPS-Std':<15} {cov_raps:<15.1%} {hard_cov_raps:<15.1%} {size_raps:<10.2f}")
    print(f"{'EntropyRAPS':<15} {cov_ent:<15.1%} {hard_cov_ent:<15.1%} {size_ent:<10.2f}")
    print("-" * 55)
    
    # Improvement
    hard_improvement = hard_cov_ent - hard_cov_raps
    print(f"\nHard-case coverage improvement: {hard_improvement:+.1%}")
    
    if hard_improvement > 0:
        print("✅ EntropyRAPS improves hard-case coverage!")
    
    print("\n" + "="*60)
    print("✅ Divergence experiment complete!")
    print("="*60)


if __name__ == "__main__":
    main()