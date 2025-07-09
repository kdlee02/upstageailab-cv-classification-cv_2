import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def calculate_class_weights(class_counts, method='inverse'):
    """
    클래스별 가중치 계산
    
    Args:
        class_counts: 각 클래스의 샘플 수 (리스트 또는 배열)
        method: 가중치 계산 방법 ('inverse', 'sqrt_inverse', 'log_inverse')
    
    Returns:
        클래스별 가중치 리스트
    """
    class_counts = np.array(class_counts)
    total_samples = class_counts.sum()
    
    if method == 'inverse':
        # 역수 기반 가중치
        weights = total_samples / (len(class_counts) * class_counts)
    elif method == 'sqrt_inverse':
        # 제곱근 역수 기반 가중치
        weights = total_samples / (len(class_counts) * np.sqrt(class_counts))
    elif method == 'log_inverse':
        # 로그 역수 기반 가중치
        weights = total_samples / (len(class_counts) * np.log(class_counts + 1))
    else:
        raise ValueError(f"지원하지 않는 가중치 계산 방법: {method}")
    
    # 가중치 정규화 (합이 클래스 수가 되도록)
    weights = weights / weights.sum() * len(class_counts)
    
    return weights.tolist()


class FocalLoss(nn.Module):
    """Focal Loss 구현"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean', class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        if self.class_weights is not None and self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """클래스별 가중치가 적용된 Focal Loss"""
    
    def __init__(self, class_weights, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # 클래스별 가중치 적용
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 