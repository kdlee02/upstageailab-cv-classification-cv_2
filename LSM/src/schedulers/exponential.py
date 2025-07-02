import torch.optim.lr_scheduler as lr_scheduler


class ExponentialScheduler:
    """Exponential LR 스캐줄러 래퍼"""
    
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
    
    def __call__(self, optimizer):
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.gamma
        ) 