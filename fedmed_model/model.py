# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from config import cfg

class ChestXRayModel(nn.Module):
    """
    ResNet-50 fine-tuned for chest X-ray classification.
    
    Architecture:
      ImageNet pretrained ResNet-50 backbone
      → Remove final FC layer
      → Add custom classifier head with dropout
      → Output: num_classes logits
    
    Why ResNet-50?
      - Strong ImageNet pretraining (feature extraction)
      - Residual connections prevent vanishing gradients
      - Well-studied for medical imaging
      - 25M parameters — powerful but not too large for FL
    """
    def __init__(self, num_classes: int = cfg.num_classes):
        super().__init__()
        
        # Load pretrained backbone
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        
        # Extract feature extractor (everything except final FC)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # Output shape: (batch, 2048, 1, 1)
        
        in_features = backbone.fc.in_features  # 2048
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (batch, 2048)
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate / 2),
            nn.Linear(128, num_classes),         # Final logits
        )
        
        # Initialize custom head weights
        self._init_weights()
        
        # Freeze backbone layers initially (unfreeze gradually)
        self._freeze_backbone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.classifier(features)
        return logits
    
    def _init_weights(self):
        """He initialization for ReLU layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _freeze_backbone(self):
        """Freeze all backbone layers. Unfreeze progressively."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen. Only classifier head training.")
    
    def unfreeze_backbone(self, layers: int = 3):
        """
        Progressively unfreeze last N layer groups.
        Call after initial warmup epochs.
        """
        children = list(self.features.children())
        # Unfreeze last `layers` children
        for child in children[-layers:]:
            for param in child.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() 
                       if p.requires_grad)
        print(f"[Model] Unfroze {layers} layers. "
              f"Trainable params: {trainable:,}")
    
    def get_parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() 
                       if p.requires_grad)
        return total, trainable


def get_model(device=None) -> ChestXRayModel:
    """Factory function — returns model on correct device."""
    if device is None:
        device = cfg.device
    model = ChestXRayModel(num_classes=cfg.num_classes)
    return model.to(device)


class FocalLoss(nn.Module):
    """
    Focal Loss — better than CrossEntropy for class imbalance.
    Down-weights easy examples (Normal cases) so model
    focuses more on hard disease cases.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma=2 is standard. alpha handles class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-ce_loss)            # probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
