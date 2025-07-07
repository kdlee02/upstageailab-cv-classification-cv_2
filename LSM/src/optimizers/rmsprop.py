import torch.optim as optim


class RMSpropOptimizer:
    """RMSprop 옵티마이저 래퍼"""
    
    def __init__(self, lr: float = 0.001, weight_decay: float = 0.0001, alpha: float = 0.99, eps: float = 1e-08, momentum: float = 0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
    
    def __call__(self, model_params):
        return optim.RMSprop(
            model_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            eps=self.eps,
            momentum=self.momentum
        ) 