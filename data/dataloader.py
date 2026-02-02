import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
import numpy as np
from typing import Tuple, Optional


class MedMNISTDataLoader:
    """Wrapper for loading MedMNIST datasets."""
    
    def __init__(self, config: dict):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']

        self.dataset_name = self.data_config['dataset']
        self.info = INFO[self.dataset_name]
        self.num_classes = len(self.info['label'])
        self.class_names = list(self.info['label'].values())

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
        
        print(f"Dataset: {self.dataset_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        
    def _get_train_transform(self) -> transforms.Compose:
        """Get training transforms with augmentation."""
        size = self.data_config['size']
        aug_config = self.data_config.get('augmentation', {})
        
        transform_list = []
        
        if aug_config.get('enabled', False):
            if aug_config.get('random_horizontal_flip', 0) > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(aug_config['random_horizontal_flip'])
                )
            
            if aug_config.get('random_rotation', 0) > 0:
                transform_list.append(
                    transforms.RandomRotation(aug_config['random_rotation'])
                )
            
            if aug_config.get('color_jitter', 0) > 0:
                jitter = aug_config['color_jitter']
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=jitter,
                        contrast=jitter,
                        saturation=jitter,
                        hue=min(jitter/2, 0.5)
                    )
                )
        
        # Resize to target size
        transform_list.append(transforms.Resize(size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))

        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        )
        
        return transforms.Compose(transform_list)

    def _get_test_transform(self) -> transforms.Compose:
        """Get test transforms without augmentation."""
        size = self.data_config['size']
        
        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ]
        
        return transforms.Compose(transform_list)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load train, validation, and test dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        DataClass = getattr(medmnist, self.info['python_class'])
        
        # Load datasets
        train_dataset = DataClass(
            split='train',
            transform=self.train_transform,
            download=self.data_config.get('download', True),
            size=self.data_config['size'],
            root=self.data_config.get('data_dir', './data')
        )
        
        val_dataset = DataClass(
            split='val',
            transform=self.test_transform,
            download=self.data_config.get('download', True),
            size=self.data_config['size'],
            root=self.data_config.get('data_dir', './data')
        )
        
        test_dataset = DataClass(
            split='test',
            transform=self.test_transform,
            download=self.data_config.get('download', True),
            size=self.data_config['size'],
            root=self.data_config.get('data_dir', './data')
        )

        batch_size = self.data_config['batch_size']
        num_workers = self.data_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def split_validation_set(self, val_dataset, 
                            pct_paramtune: float = 0.3) -> Tuple[Subset, Subset]:
        """
        Split validation set into parameter tuning and calibration sets.
        
        Args:
            val_dataset: Validation dataset
            pct_paramtune: Percentage for parameter tuning
            
        Returns:
            Tuple of (paramtune_dataset, calib_dataset)
        """
        n_total = len(val_dataset)
        n_paramtune = int(n_total * pct_paramtune)
        n_calib = n_total - n_paramtune
        
        indices = np.random.permutation(n_total)
        paramtune_indices = indices[:n_paramtune]
        calib_indices = indices[n_paramtune:]
        
        paramtune_dataset = Subset(val_dataset, paramtune_indices)
        calib_dataset = Subset(val_dataset, calib_indices)
        
        print(f"\nValidation set split for conformal prediction:")
        print(f"  Parameter tuning: {len(paramtune_dataset)} samples")
        print(f"  Calibration: {len(calib_dataset)} samples")
        
        return paramtune_dataset, calib_dataset
    
    def split_validation_3way(self, val_dataset,
                             pct_tune: float = 0.3,
                             pct_calib: float = 0.4,
                             pct_val: float = 0.3) -> Tuple[Subset, Subset, Subset]:
        """
        ✨ NEW: Three-way split for proper conformal prediction.
        
        This fixes the split protocol violation by separating:
        1. Tune set: Define entropy boundaries and scan lambda candidates
        2. Calib set: Compute quantiles for each lambda
        3. Val set: Select best lambda based on stratified coverage
        
        Args:
            val_dataset: Full validation dataset
            pct_tune: Percentage for tuning (entropy boundaries, lambda scan)
            pct_calib: Percentage for calibration (quantile computation)
            pct_val: Percentage for validation (lambda selection)
            
        Returns:
            Tuple of (tune_dataset, calib_dataset, val_dataset)
        """
        assert abs(pct_tune + pct_calib + pct_val - 1.0) < 1e-6, \
            f"Percentages must sum to 1.0, got {pct_tune + pct_calib + pct_val}"
        
        n_total = len(val_dataset)
        n_tune = int(n_total * pct_tune)
        n_calib = int(n_total * pct_calib)
        n_val = n_total - n_tune - n_calib  # Remaining samples
        
        # Random permutation for splitting
        indices = np.random.permutation(n_total)
        
        tune_indices = indices[:n_tune]
        calib_indices = indices[n_tune:n_tune+n_calib]
        val_indices = indices[n_tune+n_calib:]
        
        tune_dataset = Subset(val_dataset, tune_indices)
        calib_dataset = Subset(val_dataset, calib_indices)
        val_dataset_split = Subset(val_dataset, val_indices)
        
        print(f"\n{'='*60}")
        print("3-WAY SPLIT FOR CONFORMAL PREDICTION")
        print(f"{'='*60}")
        print(f"  Tune set (entropy boundaries):  {len(tune_dataset):>6} ({pct_tune:.1%})")
        print(f"  Calib set (quantile computation): {len(calib_dataset):>6} ({pct_calib:.1%})")
        print(f"  Val set (lambda selection):      {len(val_dataset_split):>6} ({pct_val:.1%})")
        print(f"  Total:                           {n_total:>6}")
        print(f"{'='*60}\n")
        
        return tune_dataset, calib_dataset, val_dataset_split
    
    def get_class_weights(self, dataset) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            dataset: Dataset to compute weights from
            
        Returns:
            Tensor of class weights
        """
        labels = []
        for _, label in dataset:
            labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        
        class_counts = np.bincount(labels, minlength=self.num_classes)
        
        total_samples = len(labels)
        weights = total_samples / (self.num_classes * class_counts + 1e-6)
        
        return torch.FloatTensor(weights)


def get_raw_transform(size: int = 224) -> transforms.Compose:
    """
    Get transform that returns raw images (for Grad-CAM visualization).
    
    Args:
        size: Target size
        
    Returns:
        Transform composition
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize ImageNet-normalized tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return tensor * std + mean
