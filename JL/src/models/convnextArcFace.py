import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── ArcMarginProduct (ArcFace 헤드) ───────────────────────────
class ArcMarginProduct(nn.Module):
    """
    in_features → cosine margin 적용 → scaled logits
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        # Normalize the feature vectors and weights
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        
        if not self.training or labels is None:
            return cosine * self.s
            
        # Ensure labels are within valid range
        if labels.max() >= self.out_features or labels.min() < 0:
            raise ValueError(f'Label values must be in [0, {self.out_features-1}], but got {labels.min().item()} to {labels.max().item()}')
        
        # Calculate cos(θ + m)
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Apply margin using one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to the correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


# ── ConvNeXt backbone + ArcMarginProduct ─────────────────────
class ConvNeXtArcFace(nn.Module):
    def __init__(self, cfg, s=30.0, m=0.50):
        super().__init__()
        print(f"Creating ConvNeXt + ArcFace model with {cfg.model.num_classes} classes")
        
        # Initialize backbone without the final classification layer
        self.backbone = timm.create_model(
            cfg.model.model_name,
            pretrained=cfg.model.pretrained,
            num_classes=0  # Remove the final classification layer
        )
        
        # Get feature dimension and number of classes
        in_feats = self.backbone.num_features
        self.num_classes = cfg.model.num_classes
        
        print(f"Feature dimension: {in_feats}, Number of classes: {self.num_classes}")
        
        # Initialize ArcFace head
        self.arc_head = ArcMarginProduct(
            in_features=in_feats,
            out_features=self.num_classes,
            s=s,
            m=m
        )
        
        # Freeze strategy
        if hasattr(cfg.trainer, 'freeze_epochs') and cfg.trainer.freeze_epochs > 0:
            print(f"Freezing backbone for {cfg.trainer.freeze_epochs} epochs")
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def forward(self, x, labels=None):
        # Extract features
        feats = self.backbone(x)
        
        # During training with labels, use ArcFace head
        if self.training and labels is not None:
            # Ensure labels are in the correct format
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            labels = labels.long()
            
            # Get logits from ArcFace
            logits = self.arc_head(feats, labels)
        else:
            # For validation/test or inference, compute simple cosine similarity
            logits = F.linear(F.normalize(feats), F.normalize(self.arc_head.weight))
            logits = logits * self.arc_head.s
            
        return logits

    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True