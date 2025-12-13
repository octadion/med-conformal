import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetClassifier(nn.Module):
    """ResNet18-based classifier for medical image classification."""
    
    def __init__(self, num_classes: int = 11, pretrained: bool = True,
                 dropout: float = 0.3, freeze_backbone: bool = False):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Use pretrained ImageNet weights
            dropout: Dropout rate
            freeze_backbone: Freeze backbone weights
        """
        super(ResNetClassifier, self).__init__()
        
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final classification layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        # Forward through all layers except fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_target_layer(self):
        """Get target layer for Grad-CAM."""
        return self.backbone.layer4


def get_model(config: dict, device: str = 'cuda') -> nn.Module:
    """
    Factory function to create model based on config.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        Model instance
    """
    model_config = config['model']
    
    if model_config['architecture'].lower() == 'resnet18':
        model = ResNetClassifier(
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained'],
            dropout=model_config['dropout'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    else:
        raise NotImplementedError(f"Model {model_config['architecture']} not implemented")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_config['architecture']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, 
                   device: str = 'cuda') -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")
    
    return model


def save_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                   epoch: int, metrics: dict, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Metrics dictionary
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
