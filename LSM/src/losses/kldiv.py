import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    """Kullback-Leibler Divergence 손실 함수"""
    
    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction=reduction)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 모델 출력 (log_softmax 적용 필요)
            targets: 라벨 스무딩이 적용된 타겟
        """
        # log_softmax 적용
        log_probs = F.log_softmax(predictions, dim=1)
        return self.criterion(log_probs, targets) 