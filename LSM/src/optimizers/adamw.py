import torch.optim as optim


class AdamWOptimizer:
    """AdamW 옵티마이저 래퍼"""
    
    def __init__(self, lr: float = 0.001, weight_decay: float = 0.01, betas: tuple = (0.9, 0.999)):
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
    
    def __call__(self, model_params):
        return optim.AdamW(
            model_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas
        ) 