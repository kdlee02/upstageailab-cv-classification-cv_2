import torch.optim.lr_scheduler as lr_scheduler


class PlateauScheduler:
    """ReduceLROnPlateau 스캐줄러 래퍼"""
    
    def __init__(self, mode: str = 'min', factor: float = 0.1, patience: int = 10, 
                 verbose: bool = False, threshold: float = 1e-4, threshold_mode: str = 'rel',
                 cooldown: int = 0, min_lr: float = 0, eps: float = 1e-8):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
    
    def __call__(self, optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            verbose=self.verbose,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            eps=self.eps
        ) 