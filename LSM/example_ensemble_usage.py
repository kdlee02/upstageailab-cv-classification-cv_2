#!/usr/bin/env python3
"""
앙상블 사용 예시 스크립트
"""

import torch
import torch.nn as nn
from src.utils.ensemble import EnsembleModel, TestTimeAugmentation
from src.models.resnet50 import ResNet50Model
from src.models.efficientnet import EfficientNetModel
from src.models.vit import VisionTransformerModel


def example_basic_ensemble():
    """기본 앙상블 예시"""
    print("=== 기본 앙상블 예시 ===")
    
    # 모델들 생성 (실제로는 체크포인트에서 로드)
    models = [
        ResNet50Model(num_classes=17),
        EfficientNetModel(num_classes=17),
        VisionTransformerModel(num_classes=17)
    ]
    
    # 앙상블 모델 생성
    ensemble = EnsembleModel(models, ensemble_method='averaging')
    
    # 더미 데이터로 예측
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        prediction = ensemble.predict(dummy_input, device='cpu')
        print(f"앙상블 예측 형태: {prediction.shape}")
        print(f"예측 클래스: {torch.argmax(prediction, dim=1)}")


def example_weighted_ensemble():
    """가중 앙상블 예시"""
    print("\n=== 가중 앙상블 예시 ===")
    
    # 모델들 생성
    models = [
        ResNet50Model(num_classes=17),
        EfficientNetModel(num_classes=17),
        VisionTransformerModel(num_classes=17)
    ]
    
    # 가중 앙상블 모델 생성
    ensemble = EnsembleModel(models, ensemble_method='weighted')
    
    # 가중치 설정 (검증 성능에 따라 조정)
    weights = [0.4, 0.4, 0.2]  # ResNet50, EfficientNet, ViT
    ensemble.set_weights(weights)
    
    # 더미 데이터로 예측
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        prediction = ensemble.predict(dummy_input, device='cpu')
        print(f"가중 앙상블 예측 형태: {prediction.shape}")
        print(f"예측 클래스: {torch.argmax(prediction, dim=1)}")


def example_voting_ensemble():
    """투표 앙상블 예시"""
    print("\n=== 투표 앙상블 예시 ===")
    
    # 모델들 생성
    models = [
        ResNet50Model(num_classes=17),
        EfficientNetModel(num_classes=17),
        VisionTransformerModel(num_classes=17)
    ]
    
    # 투표 앙상블 모델 생성
    ensemble = EnsembleModel(models, ensemble_method='voting')
    
    # 더미 데이터로 예측
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        prediction = ensemble.predict(dummy_input, device='cpu')
        print(f"투표 앙상블 예측 형태: {prediction.shape}")
        print(f"예측 클래스: {torch.argmax(prediction, dim=1)}")


def example_tta_ensemble():
    """TTA 앙상블 예시"""
    print("\n=== TTA 앙상블 예시 ===")
    
    # 단일 모델 생성
    model = ResNet50Model(num_classes=17)
    
    # TTA 증강 기법들 정의
    from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter
    
    augmentations = [
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=[90, 180, 270]),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    ]
    
    # TTA 모델 생성
    tta_model = TestTimeAugmentation(model, augmentations)
    
    # 더미 데이터로 예측
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        prediction = tta_model.predict(dummy_input, device='cpu')
        print(f"TTA 예측 형태: {prediction.shape}")
        print(f"예측 클래스: {torch.argmax(prediction, dim=1)}")


def example_ensemble_comparison():
    """다양한 앙상블 방법 비교"""
    print("\n=== 앙상블 방법 비교 ===")
    
    # 모델들 생성
    models = [
        ResNet50Model(num_classes=17),
        EfficientNetModel(num_classes=17),
        VisionTransformerModel(num_classes=17)
    ]
    
    # 더미 데이터
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 각 앙상블 방법별 예측
    methods = ['averaging', 'voting', 'weighted']
    
    for method in methods:
        ensemble = EnsembleModel(models, ensemble_method=method)
        
        if method == 'weighted':
            ensemble.set_weights([0.4, 0.4, 0.2])
        
        with torch.no_grad():
            prediction = ensemble.predict(dummy_input, device='cpu')
            predicted_class = torch.argmax(prediction, dim=1).item()
            confidence = torch.max(torch.softmax(prediction, dim=1), dim=1)[0].item()
            
            print(f"{method:12s}: 클래스 {predicted_class}, 신뢰도 {confidence:.4f}")


if __name__ == "__main__":
    print("앙상블 사용 예시를 실행합니다...")
    
    try:
        example_basic_ensemble()
        example_weighted_ensemble()
        example_voting_ensemble()
        example_tta_ensemble()
        example_ensemble_comparison()
        
        print("\n=== 모든 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("실제 사용 시에는 훈련된 모델의 체크포인트를 로드해야 합니다.") 