import torch
import torch.nn as nn
import torchvision.models as models


class VisionTransformerModel(nn.Module):
    """Vision Transformer 기반 문서 분류 모델"""
    
    def __init__(self, model_name: str = "vit_base_patch16_224", num_classes: int = 17, dropout_rate: float = 0.1):
        super().__init__()
        
        # Vision Transformer 백본
        if model_name == "vit_base_patch16_224":
            self.backbone = models.vit_b_16(pretrained=True)
        elif model_name == "vit_large_patch16_224":
            self.backbone = models.vit_l_16(pretrained=True)
        elif model_name == "vit_huge_patch14_224":
            self.backbone = models.vit_h_14(pretrained=True)
        else:
            raise ValueError(f"지원하지 않는 ViT 모델: {model_name}")
        
        # 마지막 분류 층을 제거
        in_features = self.backbone.heads.head.in_features
        
        # 새로운 분류 헤드
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x) 