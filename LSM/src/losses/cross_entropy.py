import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """Cross Entropy 손실 함수 래퍼"""
    
    def __init__(self, label_smoothing: float = 0.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights
        )
    
    def forward(self, predictions, targets):
        return self.criterion(predictions, targets) 