import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig, ListConfig

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='sum', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.default_reduction = reduction.lower()
        self.ignore_index = ignore_index

        if isinstance(alpha, (DictConfig, ListConfig)):
            alpha = OmegaConf.to_container(alpha, resolve=True)
        if isinstance(alpha, list):
            alpha = torch.tensor(alpha, dtype=torch.float32)
            alpha = alpha / alpha.sum()
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else None)

    def forward(self, logits, targets, reduction=None):
        reduction = (reduction or self.default_reduction).lower()
        
        if logits.dim() != 2:
            raise ValueError(f"Expected logits to be 2D (batch_size, num_classes), got {logits.dim()}D")
        if targets.dim() not in (1, 2):
            raise ValueError(f"Expected targets to be 1D or 2D, got {targets.dim()}D")
        if logits.size(0) != targets.size(0):
            raise ValueError(f"Batch size mismatch: logits {logits.size(0)} vs targets {targets.size(0)}")
        
        B, C = logits.shape

        # Handle ignore_index for hard labels
        if targets.dim() == 1:
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            logits = logits[valid_mask]
            targets = targets[valid_mask]
        elif self.ignore_index != -100:
            pass

        log_p = F.log_softmax(logits, dim=1)
        
        if targets.dim() == 1:
            log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
        else:
            # Soft labels
            if targets.size(1) != C:
                raise ValueError(f"Soft targets size mismatch: expected {C} classes, got {targets.size(1)}")
            targets = targets.float().to(logits.device)
            log_pt = (targets * log_p).sum(dim=1)
            pt = (targets * log_p.exp()).sum(dim=1)
        
        pt = pt.clamp(min=1e-8, max=1.0 - 1e-8)
        log_pt = log_pt.clamp(min=-100)

        if self.alpha is not None:
            if targets.dim() == 1:
                if len(self.alpha) != C:
                    raise ValueError(f"Alpha length {len(self.alpha)} doesn't match num_classes {C}")
                alpha_t = self.alpha[targets.long()]
            else:
                alpha_t = (targets * self.alpha.to(logits.device)).sum(dim=1)
        else:
            alpha_t = 1.0

        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if loss.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def __repr__(self):
        return (f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, "
                f"reduction='{self.default_reduction}', ignore_index={self.ignore_index})")