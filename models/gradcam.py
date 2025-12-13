import torch
import torch.nn as nn
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAMVisualizer:
    """Wrapper for Grad-CAM visualization."""
    
    def __init__(self, model: nn.Module, target_layer, device: str = 'cuda'):
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model: Model to visualize
            target_layer: Target layer for Grad-CAM (e.g., model.backbone.layer4)
            device: Device
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer]
        )
        
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class for CAM (None = predicted class)
            
        Returns:
            CAM heatmap (H, W)
        """
        # Get CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)

        return grayscale_cam[0, :]
    
    def visualize(self, input_tensor: torch.Tensor, 
                  original_image: np.ndarray,
                  target_class: Optional[int] = None,
                  colormap: str = 'jet',
                  alpha: float = 0.5) -> np.ndarray:
        """
        Create Grad-CAM visualization overlaid on original image.
        
        Args:
            input_tensor: Normalized input tensor (1, 3, H, W)
            original_image: Original RGB image (H, W, 3) in range [0, 1]
            target_class: Target class
            colormap: Colormap for heatmap
            alpha: Overlay transparency
            
        Returns:
            Visualization image
        """
        cam = self.generate_cam(input_tensor, target_class)
        
        visualization = show_cam_on_image(
            original_image, cam, use_rgb=True, 
            colormap=cv2.COLORMAP_JET
        )
        
        return visualization
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'cam'):
            del self.cam


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize ImageNet-normalized tensor to [0, 1] range for visualization.
    
    Args:
        tensor: Normalized tensor (3, H, W)
        
    Returns:
        Denormalized numpy array (H, W, 3) in range [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Denormalize
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    return img


def generate_gradcam_samples(model, conformal_model, dataloader, 
                            save_dir: Path, config: dict, 
                            class_names: List[str]):
    """
    Generate Grad-CAM visualizations for various samples.
    
    Args:
        model: Base model
        conformal_model: Conformal prediction model
        dataloader: DataLoader
        save_dir: Directory to save visualizations
        config: Configuration dictionary
        class_names: List of class names
    """
    gradcam_config = config['gradcam']

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'correct_predictions').mkdir(exist_ok=True)
    (save_dir / 'incorrect_predictions').mkdir(exist_ok=True)
    (save_dir / 'high_confidence').mkdir(exist_ok=True)
    (save_dir / 'medium_confidence').mkdir(exist_ok=True)
    (save_dir / 'low_confidence').mkdir(exist_ok=True)
    (save_dir / 'per_class').mkdir(exist_ok=True)
    
    print("\nGenerating Grad-CAM visualizations...")
    
    target_layer = model.get_target_layer()
    gradcam = GradCAMVisualizer(model, target_layer)

    correct_samples = []
    incorrect_samples = []
    high_conf_samples = []
    medium_conf_samples = []
    low_conf_samples = []
    per_class_samples = {i: [] for i in range(len(class_names))}
    
    model.eval()
    conformal_model.eval()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x_cuda = x.cuda()
            y = y.squeeze()
            
            # Get predictions
            logits, pred_sets = conformal_model(x_cuda)
            preds = torch.argmax(logits, dim=1).cpu()
            
            for i in range(x.size(0)):
                img_tensor = x[i]
                true_label = y[i].item()
                pred_label = preds[i].item()
                pred_set = pred_sets[i]
                set_size = len(pred_set)
                is_correct = pred_label == true_label
                
                sample = {
                    'image': img_tensor,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'pred_set': pred_set,
                    'set_size': set_size,
                    'correct': is_correct
                }
                
                if is_correct:
                    correct_samples.append(sample)
                else:
                    incorrect_samples.append(sample)
                
                if set_size == 1:
                    high_conf_samples.append(sample)
                elif 2 <= set_size <= 3:
                    medium_conf_samples.append(sample)
                else:
                    low_conf_samples.append(sample)

                per_class_samples[true_label].append(sample)

            n_correct = gradcam_config['n_samples_correct']
            n_incorrect = gradcam_config['n_samples_incorrect']
            if len(correct_samples) >= n_correct and len(incorrect_samples) >= n_incorrect:
                break
    
    print("  Creating visualizations...")

    _save_gradcam_grid(
        correct_samples[:gradcam_config['n_samples_correct']], 
        gradcam, class_names,
        save_dir / 'correct_predictions' / 'grid.png',
        title='Correct Predictions'
    )
    
    _save_gradcam_grid(
        incorrect_samples[:gradcam_config['n_samples_incorrect']], 
        gradcam, class_names,
        save_dir / 'incorrect_predictions' / 'grid.png',
        title='Incorrect Predictions'
    )
    
    # Confidence-based
    if len(high_conf_samples) >= 10:
        _save_gradcam_grid(
            high_conf_samples[:10], gradcam, class_names,
            save_dir / 'high_confidence' / 'grid.png',
            title='High Confidence (Set Size = 1)'
        )
    
    if len(medium_conf_samples) >= 10:
        _save_gradcam_grid(
            medium_conf_samples[:10], gradcam, class_names,
            save_dir / 'medium_confidence' / 'grid.png',
            title='Medium Confidence (Set Size 2-3)'
        )
    
    if len(low_conf_samples) >= 10:
        _save_gradcam_grid(
            low_conf_samples[:10], gradcam, class_names,
            save_dir / 'low_confidence' / 'grid.png',
            title='Low Confidence (Set Size 4+)'
        )
    
    n_per_class = gradcam_config.get('n_samples_per_class', 2)
    for class_id, class_name in enumerate(class_names):
        if len(per_class_samples[class_id]) >= n_per_class:
            _save_gradcam_grid(
                per_class_samples[class_id][:n_per_class], 
                gradcam, class_names,
                save_dir / 'per_class' / f'{class_name}.png',
                title=f'Class: {class_name}',
                ncols=n_per_class
            )
    
    print("  ✓ Grad-CAM visualizations saved!")
    
    del gradcam


def _save_gradcam_grid(samples: List[dict], gradcam: GradCAMVisualizer,
                       class_names: List[str], save_path: Path,
                       title: str, ncols: int = 5):
    """
    Save a grid of Grad-CAM visualizations.
    
    Args:
        samples: List of sample dictionaries
        gradcam: GradCAM visualizer
        class_names: List of class names
        save_path: Path to save figure
        title: Figure title
        ncols: Number of columns in grid
    """
    n_samples = len(samples)
    nrows = (n_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Get image
        img_tensor = sample['image'].unsqueeze(0).cuda()
        original_img = denormalize_image(sample['image'])
        
        # Generate Grad-CAM
        cam_img = gradcam.visualize(img_tensor, original_img)
        
        ax.imshow(cam_img)
        ax.axis('off')
        
        true_name = class_names[sample['true_label']]
        pred_name = class_names[sample['pred_label']]
        set_size = sample['set_size']
        
        if sample['correct']:
            color = 'green'
            subtitle = f"✓ {true_name}\nSet: {set_size}"
        else:
            color = 'red'
            subtitle = f"True: {true_name}\nPred: {pred_name}\nSet: {set_size}"
        
        ax.set_title(subtitle, fontsize=8, color=color, weight='bold')

    for idx in range(n_samples, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
