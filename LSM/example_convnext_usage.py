#!/usr/bin/env python3
"""
ConvNeXt 모델 사용 예제 스크립트

이 스크립트는 ConvNeXt 모델의 다양한 변형을 사용하는 방법을 보여줍니다.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.convnext import (
    ConvNeXtModel, 
    ConvNeXtWithAttention, 
    ConvNeXtWithFocalLoss,
    create_convnext_model
)
from src.lightning_modules.classification_module import ClassificationModule


def test_convnext_models():
    """다양한 ConvNeXt 모델 테스트"""
    print("=== ConvNeXt 모델 테스트 ===")
    
    # 테스트용 입력 데이터
    batch_size = 4
    num_classes = 17
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # 1. 기본 ConvNeXt 모델들
    print("\n1. 기본 ConvNeXt 모델들:")
    
    convnext_models = [
        ('convnext_tiny', 'convnext_tiny'),
        ('convnext_small', 'convnext_small'),
        ('convnext_base', 'convnext_base'),
        ('convnext_large', 'convnext_large')
    ]
    
    for model_name, model_type in convnext_models:
        try:
            model = ConvNeXtModel(
                model_name=model_type,
                num_classes=num_classes,
                dropout_rate=0.1,
                pretrained=False  # 테스트를 위해 pretrained=False
            )
            
            with torch.no_grad():
                output = model(input_tensor)
            
            print(f"  {model_name}: 입력 {input_tensor.shape} -> 출력 {output.shape}")
            
        except Exception as e:
            print(f"  {model_name}: 오류 - {e}")
    
    # 2. 어텐션 메커니즘이 추가된 ConvNeXt
    print("\n2. 어텐션 메커니즘이 추가된 ConvNeXt:")
    try:
        attention_model = ConvNeXtWithAttention(
            model_name='convnext_tiny',
            num_classes=num_classes,
            dropout_rate=0.1,
            attention_dim=256,
            pretrained=False
        )
        
        with torch.no_grad():
            output = attention_model(input_tensor)
        
        print(f"  ConvNeXt + Attention: 입력 {input_tensor.shape} -> 출력 {output.shape}")
        
    except Exception as e:
        print(f"  ConvNeXt + Attention: 오류 - {e}")
    
    # 3. Focal Loss 최적화 ConvNeXt
    print("\n3. Focal Loss 최적화 ConvNeXt:")
    try:
        focal_model = ConvNeXtWithFocalLoss(
            model_name='convnext_tiny',
            num_classes=num_classes,
            dropout_rate=0.1,
            alpha=1.0,
            gamma=2.0,
            pretrained=False
        )
        
        with torch.no_grad():
            output = focal_model(input_tensor)
        
        # Focal Loss 계산 테스트
        targets = torch.randint(0, num_classes, (batch_size,))
        focal_loss = focal_model.focal_loss(output, targets)
        
        print(f"  ConvNeXt + Focal Loss: 입력 {input_tensor.shape} -> 출력 {output.shape}")
        print(f"  Focal Loss 값: {focal_loss.item():.4f}")
        
    except Exception as e:
        print(f"  ConvNeXt + Focal Loss: 오류 - {e}")
    
    # 4. 팩토리 함수 사용
    print("\n4. 팩토리 함수 사용:")
    try:
        factory_model = create_convnext_model(
            model_name='convnext_tiny',
            num_classes=num_classes,
            dropout_rate=0.1,
            use_attention=True,
            use_focal_loss=False,
            pretrained=False,
            attention_dim=256
        )
        
        with torch.no_grad():
            output = factory_model(input_tensor)
        
        print(f"  팩토리 함수 (Attention): 입력 {input_tensor.shape} -> 출력 {output.shape}")
        
    except Exception as e:
        print(f"  팩토리 함수: 오류 - {e}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """메인 함수 - Hydra 설정을 사용한 ConvNeXt 실험"""
    
    print("=== ConvNeXt 모델 실험 시작 ===")
    
    # 설정을 딕셔너리로 변환
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 모델 설정을 ConvNeXt로 변경
    config_dict['model']['name'] = 'convnext'
    
    # 클래스명 설정
    class_names = [f"class_{i}" for i in range(cfg.num_classes)]
    
    # ClassificationModule 생성
    model = ClassificationModule(config_dict)
    model.set_class_names(class_names)
    
    print(f"모델 타입: {type(model.model)}")
    print(f"모델 설정: {config_dict['model']}")
    
    # 테스트용 입력
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # 모델 테스트
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"입력 형태: {input_tensor.shape}")
    print(f"출력 형태: {output.shape}")
    print(f"예측 클래스: {output.argmax(dim=1)}")
    
    print("\n=== ConvNeXt 모델 실험 완료 ===")


def compare_convnext_variants():
    """ConvNeXt 변형 모델들 비교"""
    print("\n=== ConvNeXt 변형 모델들 비교 ===")
    
    batch_size = 4
    num_classes = 17
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # 모델 변형들
    variants = [
        {
            'name': 'ConvNeXt Tiny (기본)',
            'config': {
                'model_name': 'convnext_tiny',
                'use_attention': False,
                'use_focal_loss': False
            }
        },
        {
            'name': 'ConvNeXt Small',
            'config': {
                'model_name': 'convnext_small',
                'use_attention': False,
                'use_focal_loss': False
            }
        },
        {
            'name': 'ConvNeXt Base',
            'config': {
                'model_name': 'convnext_base',
                'use_attention': False,
                'use_focal_loss': False
            }
        },
        {
            'name': 'ConvNeXt Tiny + Attention',
            'config': {
                'model_name': 'convnext_tiny',
                'use_attention': True,
                'use_focal_loss': False,
                'attention_dim': 256
            }
        },
        {
            'name': 'ConvNeXt Tiny + Focal Loss',
            'config': {
                'model_name': 'convnext_tiny',
                'use_attention': False,
                'use_focal_loss': True,
                'alpha': 1.0,
                'gamma': 2.0
            }
        }
    ]
    
    results = []
    
    for variant in variants:
        try:
            model = create_convnext_model(
                num_classes=num_classes,
                dropout_rate=0.1,
                pretrained=False,
                **variant['config']
            )
            
            # 모델 파라미터 수 계산
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 추론 시간 측정
            model.eval()
            with torch.no_grad():
                import time
                start_time = time.time()
                output = model(input_tensor)
                inference_time = time.time() - start_time
            
            results.append({
                'name': variant['name'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                'inference_time': inference_time,
                'output_shape': output.shape,
                'status': '성공'
            })
            
        except Exception as e:
            results.append({
                'name': variant['name'],
                'total_params': 0,
                'trainable_params': 0,
                'inference_time': 0,
                'output_shape': None,
                'status': f'오류: {e}'
            })
    
    # 결과 출력
    print(f"{'모델명':<25} {'총 파라미터':<12} {'학습 파라미터':<12} {'추론 시간(ms)':<12} {'상태':<10}")
    print("-" * 80)
    
    for result in results:
        if result['status'] == '성공':
            print(f"{result['name']:<25} {result['total_params']:<12,} {result['trainable_params']:<12,} "
                  f"{result['inference_time']*1000:<12.2f} {result['status']:<10}")
        else:
            print(f"{result['name']:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {result['status']:<10}")


if __name__ == '__main__':
    # 기본 모델 테스트
    test_convnext_models()
    
    # 변형 모델들 비교
    compare_convnext_variants()
    
    # Hydra 설정을 사용한 실험 (선택사항)
    # main() 