import torch.optim as optim


class SGDOptimizer:
    """SGD 옵티마이저 래퍼"""
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0001):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
    
    def __call__(self, model_params):
        return optim.SGD(
            model_params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        ) 