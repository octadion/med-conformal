import torch
import yaml
import sys
from pathlib import Path
sys.path.append('.')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model
from models.conformal_wrapper import SizeOptimizedRAPS, EntropyStratifiedRAPS, MondrianCP
from training.trainer import Trainer
import numpy as np
from scipy.stats import entropy

class ImbalancedDifficultyDataset:
    """
    Create imbalanced difficulty distribution from existing dataset.
    
    Strategy:
    1. Compute entropy for all samples (using trained model)
    2. Define difficulty strata (tertiles)
    3. Subsample to create imbalance:
       - Easy: Keep 80-90%
       - Medium: Keep 15-20%
       - Hard: Keep 5-10%
    
    Args:
        base_dataset: Original dataset (e.g., PathMNIST validation set)
        model: Trained model (to compute entropy)
        hard_ratio: Fraction of hard cases to keep (default 0.1)
        medium_ratio: Fraction of medium cases to keep (default 0.2)
        easy_ratio: Fraction of easy cases to keep (default 0.9)
        device: 'cuda' or 'cpu'
        seed: Random seed for reproducibility
    """
    
    def __init__(self, base_dataset, model, 
                 hard_ratio=0.1, medium_ratio=0.2, easy_ratio=0.9,
                 device='cuda', seed=42):
        
        self.base_dataset = base_dataset
        self.model = model
        self.device = device
        
        print(f"[Imbalanced Dataset] Creating difficulty imbalance...")
        print(f"   Target ratios - Easy: {easy_ratio:.0%}, Medium: {medium_ratio:.0%}, Hard: {hard_ratio:.0%}")
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Step 1: Compute entropy for all samples
        entropies = self._compute_entropy_for_dataset()
        
        # Step 2: Define strata
        t1, t2 = np.quantile(entropies, [0.33, 0.66])
        
        easy_mask = entropies <= t1
        medium_mask = (entropies > t1) & (entropies <= t2)
        hard_mask = entropies > t2
        
        easy_indices = np.where(easy_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        hard_indices = np.where(hard_mask)[0]
        
        print(f"   Original counts - Easy: {len(easy_indices)}, Medium: {len(medium_indices)}, Hard: {len(hard_indices)}")
        
        # Step 3: Subsample each stratum
        n_easy = int(len(easy_indices) * easy_ratio)
        n_medium = int(len(medium_indices) * medium_ratio)
        n_hard = int(len(hard_indices) * hard_ratio)
        
        selected_easy = np.random.choice(easy_indices, n_easy, replace=False)
        selected_medium = np.random.choice(medium_indices, n_medium, replace=False)
        selected_hard = np.random.choice(hard_indices, n_hard, replace=False)
        
        # Combine and shuffle
        self.selected_indices = np.concatenate([selected_easy, selected_medium, selected_hard])
        np.random.shuffle(self.selected_indices)
        
        # Store strata information for analysis
        self.easy_mask_new = np.isin(self.selected_indices, selected_easy)
        self.medium_mask_new = np.isin(self.selected_indices, selected_medium)
        self.hard_mask_new = np.isin(self.selected_indices, selected_hard)
        
        print(f"   Imbalanced counts - Easy: {n_easy}, Medium: {n_medium}, Hard: {n_hard}")
        print(f"   New distribution - Easy: {n_easy/(n_easy+n_medium+n_hard):.1%}, "
              f"Medium: {n_medium/(n_easy+n_medium+n_hard):.1%}, "
              f"Hard: {n_hard/(n_easy+n_medium+n_hard):.1%}")
    
    def _compute_entropy_for_dataset(self):
        """Compute entropy for all samples in dataset."""
        self.model.eval()
        entropies = []
        
        # Create temporary loader
        loader = torch.utils.data.DataLoader(
            self.base_dataset, 
            batch_size=128, 
            shuffle=False
        )
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                batch_entropy = entropy(probs, axis=1)
                entropies.extend(batch_entropy)
        
        return np.array(entropies)
    
    def get_subset(self):
        """Return PyTorch Subset with selected indices."""
        return Subset(self.base_dataset, self.selected_indices)
    
    def get_stratum_info(self):
        """Return dictionary with stratum information for analysis."""
        return {
            'easy_indices': np.where(self.easy_mask_new)[0],
            'medium_indices': np.where(self.medium_mask_new)[0],
            'hard_indices': np.where(self.hard_mask_new)[0],
            'easy_count': self.easy_mask_new.sum(),
            'medium_count': self.medium_mask_new.sum(),
            'hard_count': self.hard_mask_new.sum(),
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_imbalanced_splits(dataset, model, config, device='cuda'):
    """
    High-level wrapper to create imbalanced cal/tune/test splits.
    
    Usage in main.py:
        if args.imbalanced:
            val_dataset, test_dataset = create_imbalanced_splits(
                dataset, model, config, device
            )
    
    Returns:
        imbalanced_val: Imbalanced validation set (for cal/tune split)
        imbalanced_test: Imbalanced test set
        strata_info: Dict with stratum information
    """
    
    # Create imbalanced version of validation set
    print("\n[Creating Imbalanced Validation Set]")
    imb_val = ImbalancedDifficultyDataset(
        dataset.val_dataset,
        model,
        hard_ratio=0.1,   # Only 10% of hard cases
        medium_ratio=0.2, # 20% of medium
        easy_ratio=0.9,   # 90% of easy
        device=device,
        seed=config.get('seed', 42)
    )
    
    # Create imbalanced version of test set
    print("\n[Creating Imbalanced Test Set]")
    imb_test = ImbalancedDifficultyDataset(
        dataset.test_dataset,
        model,
        hard_ratio=0.1,
        medium_ratio=0.2,
        easy_ratio=0.9,
        device=device,
        seed=config.get('seed', 42) + 1  # Different seed for test
    )
    
    return imb_val.get_subset(), imb_test.get_subset(), {
        'val_strata': imb_val.get_stratum_info(),
        'test_strata': imb_test.get_stratum_info()
    }

def main():
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['dataset'] = 'pathmnist'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    data_loader = MedMNISTDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    # Train model
    print("[1/4] Training model...")
    model = get_model(config, device)
    trainer = Trainer(model, config, device)
    trained_model = trainer.train(train_loader, val_loader, None)
    
    # Create IMBALANCED dataset
    print("[2/4] Creating imbalanced difficulty dataset...")
    imb_val = ImbalancedDifficultyDataset(
        val_loader.dataset, trained_model,
        hard_ratio=0.1, medium_ratio=0.2, easy_ratio=0.9,
        device=device, seed=42
    )
    imb_test = ImbalancedDifficultyDataset(
        test_loader.dataset, trained_model,
        hard_ratio=0.1, medium_ratio=0.2, easy_ratio=0.9,
        device=device, seed=43
    )
    
    # Split to cal/tune
    imb_val_ds = imb_val.get_subset()
    n = len(imb_val_ds)
    n_cal = int(0.7 * n)
    cal_ds, tune_ds = torch.utils.data.random_split(imb_val_ds, [n_cal, n - n_cal])
    
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=32, shuffle=False)
    tune_loader = torch.utils.data.DataLoader(tune_ds, batch_size=32, shuffle=False)
    test_imb_loader = torch.utils.data.DataLoader(imb_test.get_subset(), batch_size=32, shuffle=False)
    
    # Run methods
    print("[3/4] Running conformal methods...")
    raps = SizeOptimizedRAPS(trained_model, cal_loader, tune_loader, alpha=0.1, k_reg=2, device=device)
    entropy_raps = EntropyStratifiedRAPS(trained_model, cal_loader, tune_loader, alpha=0.1, k_reg=2, device=device)
    
    print(f"\n{'='*60}")
    print("DIVERGENCE RESULTS:")
    print(f"{'='*60}")
    print(f"Standard RAPS λ: {raps.lamda_star:.5f}")
    print(f"EntropyRAPS λ: {entropy_raps.lamda_star:.5f}")
    print(f"Difference: {abs(raps.lamda_star - entropy_raps.lamda_star):.5f}")
    
    if abs(raps.lamda_star - entropy_raps.lamda_star) > 0.01:
        print("\n✅ DIVERGENCE DETECTED!")
    else:
        print("\n⚠️ No divergence (try more extreme imbalance)")
    
    # Evaluate coverage
    print("\n[4/4] Evaluating coverage...")
    from models.conformal_wrapper import predict_with_sets
    
    # RAPS
    _, _, probs_raps, sets_raps, labels = predict_with_sets(raps, test_imb_loader)
    cov_raps = np.mean([labels[i] in sets_raps[i] for i in range(len(labels))])
    
    # EntropyRAPS
    _, _, probs_ent, sets_ent, labels = predict_with_sets(entropy_raps, test_imb_loader)
    cov_ent = np.mean([labels[i] in sets_ent[i] for i in range(len(labels))])
    
    # Hard-case coverage
    ent_vals = entropy(probs_raps, axis=1)
    hard_mask = ent_vals > np.quantile(ent_vals, 0.9)  # Top 10% hard
    
    hard_cov_raps = np.mean([labels[i] in sets_raps[i] for i in range(len(labels)) if hard_mask[i]])
    hard_cov_ent = np.mean([labels[i] in sets_ent[i] for i in range(len(labels)) if hard_mask[i]])
    
    print(f"\nStandard RAPS: Overall={cov_raps:.1%}, Hard={hard_cov_raps:.1%}")
    print(f"EntropyRAPS: Overall={cov_ent:.1%}, Hard={hard_cov_ent:.1%}")
    
    print("\n✅ Divergence experiment complete!")

if __name__ == "__main__":
    main()