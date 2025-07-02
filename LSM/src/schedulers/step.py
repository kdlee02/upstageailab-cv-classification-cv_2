import torch.optim.lr_scheduler as lr_scheduler


class StepScheduler:
    """Step LR 스캐줄러 래퍼"""
    
    def __init__(self, step_size: int = 30, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma
    
    def __call__(self, optimizer):
        return lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        ) 