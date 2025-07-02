import torch.optim.lr_scheduler as lr_scheduler


class CosineScheduler:
    """Cosine Annealing 스캐줄러 래퍼"""
    
    def __init__(self, T_max: int = 100, eta_min: float = 0.0):
        self.T_max = T_max
        self.eta_min = eta_min
    
    def __call__(self, optimizer):
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min
        ) 