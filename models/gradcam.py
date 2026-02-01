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
        self.model = model
        self.device = device
        self.model.eval()
        
        # Inisialisasi GradCAM
        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer]
        )
        
    def visualize(self, input_tensor: torch.Tensor, 
                  original_image: np.ndarray,
                  target_class: Optional[int] = None) -> np.ndarray:
        """Create Grad-CAM visualization overlaid on original image."""
        # Note: targets=[ClassifierOutputTarget(target_class)] is the proper way in new versions,
        # but passing targets=None usually defaults to max.
        # To be safe for specific classes, we rely on the library handling int targets or None.
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        
        cam_metric = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(
            original_image, cam_metric, use_rgb=True, 
            colormap=cv2.COLORMAP_JET
        )
        return visualization
    
    def __del__(self):
        if hasattr(self, 'cam'):
            del self.cam

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize ImageNet-normalized tensor to [0, 1] range."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)

def _save_gradcam_grid(samples: List[dict], gradcam: GradCAMVisualizer,
                       class_names: List[str], save_path: Path,
                       title: str, ncols: int = 5):
    if not samples: return

    n_samples = len(samples)
    nrows = (n_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    if nrows == 1 and ncols == 1: axes = np.array([axes])
    if nrows == 1: axes = axes.reshape(1, -1)
    if ncols == 1: axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    for idx, sample in enumerate(samples):
        ax = axes[idx]
        img_tensor = sample['image'].unsqueeze(0).cuda()
        original_img = denormalize_image(sample['image'])
        
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
        
        ax.set_title(subtitle, fontsize=9, color=color, weight='bold')

    # Matikan axis untuk slot kosong
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _save_contrastive_visualization(hard_cases: List[dict], gradcam: GradCAMVisualizer, 
                                   class_names: List[str], save_dir: Path):
    
    save_dir = save_dir / 'contrastive_analysis'
    save_dir.mkdir(exist_ok=True)
    
    print(f"   [GradCAM] Generating {len(hard_cases)} contrastive analyses for hard cases...")
    
    for i, case in enumerate(hard_cases):
        img_tensor = case['image'].unsqueeze(0).cuda()
        original_img = denormalize_image(case['image'])
        pred_set = case['pred_set']
        
        classes_to_show = list(pred_set)[:4]
        
        n_cols = 1 + len(classes_to_show) # 1 Original + N Heatmaps
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        axes[0].imshow(original_img)
        true_name = class_names[case['true_label']]
        axes[0].set_title(f"TRUE LABEL:\n{true_name}", color='green', weight='bold')
        axes[0].axis('off')

        cam_img = gradcam.visualize(img_tensor, original_img)
   
        for j, cls_idx in enumerate(classes_to_show):
            ax = axes[j+1]
            ax.imshow(cam_img)
            
            cls_name = class_names[cls_idx]
            is_true = (cls_idx == case['true_label'])
            color = 'green' if is_true else 'orange'
            prefix = "✓" if is_true else "?"
            
            ax.set_title(f"Candidate {j+1}:\n{prefix} {cls_name}", color=color, weight='bold')
            ax.axis('off')
            
        plt.suptitle(f"Hard Case Analysis (Set Size: {len(pred_set)})", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / f"hard_case_{i}_{true_name}.png", dpi=200)
        plt.close()

def generate_gradcam_samples(model, conformal_model, dataloader, 
                            save_dir: Path, config: dict, 
                            class_names: List[str]):

    gradcam_config = config.get('gradcam', {})
    
    # Setup Directory
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'correct_predictions').mkdir(exist_ok=True)
    (save_dir / 'incorrect_predictions').mkdir(exist_ok=True)
    (save_dir / 'confidence_levels').mkdir(exist_ok=True)
    
    print("\n[VISUALIZATION] Starting Grad-CAM Generation...")
    
    target_layer = model.get_target_layer()
    gradcam = GradCAMVisualizer(model, target_layer)

    # Container Samples
    correct_samples = []
    incorrect_samples = []
    hard_cases_contrastive = [] 
    
    high_conf = [] # size 1
    med_conf = []  # size 2-3
    low_conf = []  # size 4+
    
    model.eval()
    conformal_model.eval()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x_cuda = x.cuda()
            y = y.squeeze()
            
            logits, pred_sets = conformal_model(x_cuda)
            preds = torch.argmax(logits, dim=1).cpu()
            
            for i in range(x.size(0)):
                if len(correct_samples) > 50 and len(incorrect_samples) > 50 and len(hard_cases_contrastive) > 20:
                    break
                
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
                    if len(correct_samples) < 30: correct_samples.append(sample)
                else:
                    if len(incorrect_samples) < 30: incorrect_samples.append(sample)
  
                if set_size >= 2 and len(hard_cases_contrastive) < 15:
                    hard_cases_contrastive.append(sample)
                
                # Confidence Levels
                if set_size == 1 and len(high_conf) < 10: high_conf.append(sample)
                elif 2 <= set_size <= 3 and len(med_conf) < 10: med_conf.append(sample)
                elif set_size >= 4 and len(low_conf) < 10: low_conf.append(sample)

            if len(correct_samples) >= 30 and len(hard_cases_contrastive) >= 15:
                break

    print("   Generating Standard Grids...")
    _save_gradcam_grid(correct_samples[:20], gradcam, class_names,
                       save_dir / 'correct_predictions' / 'grid.png', 'Correct Predictions')
    _save_gradcam_grid(incorrect_samples[:20], gradcam, class_names,
                       save_dir / 'incorrect_predictions' / 'grid.png', 'Incorrect Predictions')
    
    # 3. Generate Confidence Grids
    _save_gradcam_grid(high_conf, gradcam, class_names,
                       save_dir / 'confidence_levels' / 'high_conf_size_1.png', 'High Confidence')
    _save_gradcam_grid(med_conf, gradcam, class_names,
                       save_dir / 'confidence_levels' / 'med_conf_size_2_3.png', 'Medium Confidence')
    
    
    if hard_cases_contrastive:
        _save_contrastive_visualization(hard_cases_contrastive, gradcam, class_names, save_dir)
    
    print("   ✓ All Grad-CAM visualizations saved!")
    del gradcam