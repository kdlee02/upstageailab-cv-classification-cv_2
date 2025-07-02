import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Model(nn.Module):
    """ResNet50 기반 문서 분류 모델"""
    
    def __init__(self, pretrained: bool = True, num_classes: int = 17, dropout_rate: float = 0.5):
        super().__init__()
        
        # ResNet50 백본
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 마지막 fully connected 층을 제거
        in_features = self.backbone.fc.in_features
        
        # 새로운 분류 헤드
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x) 