"""
ConvNeXt 모델 구현

ConvNeXt는 Vision Transformer의 설계 원칙을 CNN에 적용한 모델입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large


class ConvNeXtModel(nn.Module):
    """ConvNeXt 모델 래퍼 클래스"""
    
    def __init__(self, model_name='convnext_tiny', num_classes=17, dropout_rate=0.1, pretrained=True):
        """
        ConvNeXt 모델 초기화
        
        Args:
            model_name: 모델 크기 ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large')
            num_classes: 분류할 클래스 수
            dropout_rate: 드롭아웃 비율
            pretrained: 사전 훈련된 가중치 사용 여부
        """
        super().__init__()
        
        # 모델 이름에 따른 ConvNeXt 모델 선택
        if model_name == 'convnext_tiny':
            self.backbone = convnext_tiny(pretrained=pretrained)
            feature_dim = 768
        elif model_name == 'convnext_small':
            self.backbone = convnext_small(pretrained=pretrained)
            feature_dim = 768
        elif model_name == 'convnext_base':
            self.backbone = convnext_base(pretrained=pretrained)
            feature_dim = 1024
        elif model_name == 'convnext_large':
            self.backbone = convnext_large(pretrained=pretrained)
            feature_dim = 1536
        else:
            raise ValueError(f"지원하지 않는 ConvNeXt 모델: {model_name}")
        
        # 분류 헤드 수정
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.num_classes = num_classes
        self.model_name = model_name
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
        
        Returns:
            분류 로짓 (B, num_classes)
        """
        return self.backbone(x)
    
    def get_feature_dim(self):
        """특성 차원 반환"""
        if self.model_name in ['convnext_tiny', 'convnext_small']:
            return 768
        elif self.model_name == 'convnext_base':
            return 1024
        elif self.model_name == 'convnext_large':
            return 1536
        else:
            raise ValueError(f"알 수 없는 모델: {self.model_name}")


class ConvNeXtWithAttention(nn.Module):
    """어텐션 메커니즘을 추가한 ConvNeXt 모델"""
    
    def __init__(self, model_name='convnext_tiny', num_classes=17, dropout_rate=0.1, 
                 attention_dim=256, pretrained=True):
        """
        어텐션을 추가한 ConvNeXt 모델 초기화
        
        Args:
            model_name: 모델 크기
            num_classes: 분류할 클래스 수
            dropout_rate: 드롭아웃 비율
            attention_dim: 어텐션 차원
            pretrained: 사전 훈련된 가중치 사용 여부
        """
        super().__init__()
        
        # 기본 ConvNeXt 모델
        self.convnext = ConvNeXtModel(model_name, num_classes, dropout_rate, pretrained)
        
        # 어텐션 메커니즘
        feature_dim = self.convnext.get_feature_dim()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 어텐션 후 분류 헤드
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
        
        Returns:
            분류 로짓 (B, num_classes)
        """
        # ConvNeXt 백본에서 특성 추출
        features = self.convnext.backbone.forward_features(x)
        
        # 특성을 어텐션에 맞는 형태로 변환
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 어텐션 적용
        attended_features, _ = self.attention(features, features, features)
        
        # 글로벌 평균 풀링
        global_features = attended_features.mean(dim=1)  # (B, C)
        
        # 분류
        output = self.classifier(global_features)
        
        return output


class ConvNeXtWithFocalLoss(nn.Module):
    """Focal Loss에 최적화된 ConvNeXt 모델"""
    
    def __init__(self, model_name='convnext_tiny', num_classes=17, dropout_rate=0.1, 
                 alpha=1.0, gamma=2.0, pretrained=True):
        """
        Focal Loss 최적화 ConvNeXt 모델 초기화
        
        Args:
            model_name: 모델 크기
            num_classes: 분류할 클래스 수
            dropout_rate: 드롭아웃 비율
            alpha: Focal Loss의 alpha 파라미터
            gamma: Focal Loss의 gamma 파라미터
            pretrained: 사전 훈련된 가중치 사용 여부
        """
        super().__init__()
        
        self.convnext = ConvNeXtModel(model_name, num_classes, dropout_rate, pretrained)
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
        
        Returns:
            분류 로짓 (B, num_classes)
        """
        return self.convnext(x)
    
    def focal_loss(self, outputs, targets):
        """
        Focal Loss 계산
        
        Args:
            outputs: 모델 출력 (B, num_classes)
            targets: 타겟 레이블 (B,)
        
        Returns:
            Focal Loss 값
        """
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_convnext_model(model_name='convnext_tiny', num_classes=17, dropout_rate=0.1, 
                         use_attention=False, use_focal_loss=False, pretrained=True, **kwargs):
    """
    ConvNeXt 모델 생성 팩토리 함수
    
    Args:
        model_name: 모델 크기
        num_classes: 분류할 클래스 수
        dropout_rate: 드롭아웃 비율
        use_attention: 어텐션 메커니즘 사용 여부
        use_focal_loss: Focal Loss 최적화 사용 여부
        pretrained: 사전 훈련된 가중치 사용 여부
        **kwargs: 추가 파라미터
    
    Returns:
        ConvNeXt 모델 인스턴스
    """
    if use_attention:
        attention_dim = kwargs.get('attention_dim', 256)
        return ConvNeXtWithAttention(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            attention_dim=attention_dim,
            pretrained=pretrained
        )
    elif use_focal_loss:
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return ConvNeXtWithFocalLoss(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            alpha=alpha,
            gamma=gamma,
            pretrained=pretrained
        )
    else:
        return ConvNeXtModel(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        ) 