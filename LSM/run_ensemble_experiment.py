#!/usr/bin/env python3
"""
앙상블 실험 실행 스크립트
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import wandb
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import create_data_loaders
from src.utils.ensemble import EnsembleModel, TestTimeAugmentation
from src.utils.prediction import predict_with_ensemble, save_ensemble_predictions
from src.utils.env_utils import check_required_env_vars, get_wandb_config
from src.utils.auto_registry import create_model


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """앙상블 실험 메인 함수"""
    
    print("=== 앙상블 실험 시작 ===")
    print(f"앙상블 방법: {cfg.ensemble.method}")
    print(f"모델 수: {len(cfg.ensemble.models)}")
    
    # 환경 변수 확인
    check_required_env_vars()
    
    # 시드 설정
    pl.seed_everything(cfg.seed)
    
    # 디렉터리 생성
    experiment_dir = Path(cfg.save.model_dir) / f"{cfg.experiment.name}_ensemble"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 설정 가져오기
    wandb_config = get_wandb_config()
    
    # wandb 초기화
    wandb_logger = wandb.init(
        project=wandb_config['project'],
        name=f"{cfg.experiment.name}_ensemble",
        tags=cfg.experiment.tags + ["ensemble"],
        config=OmegaConf.to_container(cfg.ensemble, resolve=True)
    )
    
    # 데이터 로더 생성
    print("데이터 로더 생성 중...")
    data_loaders = create_data_loaders(cfg)
    test_loader = data_loaders['test_loader']
    class_names = data_loaders['class_names']
    
    print(f"테스트 샘플 수: {len(test_loader.dataset)}")
    print(f"클래스 수: {len(class_names)}")
    
    # 모델 설정 준비
    model_configs = []
    weights = []
    
    for model_config in cfg.ensemble.models:
        model_name = model_config.name
        checkpoint_path = model_config.checkpoint_path
        weight = model_config.weight
        
        print(f"모델 로드 중: {model_name}")
        print(f"체크포인트: {checkpoint_path}")
        
        # 모델 클래스 결정 (레지스트리 시스템 사용)
        # 모델 이름에서 타입 추출 (예: "resnet50_model" -> "resnet50")
        model_type = model_name.lower().split('_')[0]  # 첫 번째 부분을 모델 타입으로 사용
        
        # 모델 설정 생성
        model_cfg = {
            '_target_': cfg.model._target_,
            **{k: v for k, v in cfg.model.items() if k != '_target_'}
        }
        
        # 모델 클래스 가져오기 (실제 인스턴스는 생성하지 않음)
        from src.utils.auto_registry import MODEL_AUTO_REGISTRY
        model_class = MODEL_AUTO_REGISTRY.get(model_type)
        model_params = {k: v for k, v in cfg.model.items() if k != '_target_'}
        
        model_configs.append({
            'model_class': model_class,
            'model_params': model_params,
            'checkpoint_path': checkpoint_path
        })
        weights.append(weight)
    
    # 앙상블 예측 수행
    print("앙상블 예측 수행 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    filenames, predictions = predict_with_ensemble(
        model_configs=model_configs,
        test_loader=test_loader,
        device=device,
        ensemble_method=cfg.ensemble.method,
        weights=weights if cfg.ensemble.method == 'weighted' else None
    )
    
    # 예측 결과 저장
    predictions_path = experiment_dir / "ensemble_predictions.csv"
    ensemble_config = {
        'method': cfg.ensemble.method,
        'num_models': len(cfg.ensemble.models),
        'weights': weights if cfg.ensemble.method == 'weighted' else None,
        'models': [model.name for model in cfg.ensemble.models]
    }
    
    save_ensemble_predictions(
        filenames=filenames,
        predictions=predictions,
        ensemble_config=ensemble_config,
        output_path=str(predictions_path),
        class_names=class_names,
        use_sample_order=True
    )
    
    # TTA가 활성화된 경우 추가 예측
    if cfg.ensemble.tta.enabled:
        print("TTA 예측 수행 중...")
        
        # TTA 증강 기법 생성
        augmentations = []
        for aug_config in cfg.ensemble.tta.augmentations:
            aug_type = aug_config.type
            
            if aug_type == "horizontal_flip":
                from torchvision.transforms import RandomHorizontalFlip
                aug = RandomHorizontalFlip(p=aug_config.probability)
            elif aug_type == "rotation":
                from torchvision.transforms import RandomRotation
                aug = RandomRotation(degrees=aug_config.degrees)
            elif aug_type == "color_jitter":
                from torchvision.transforms import ColorJitter
                aug = ColorJitter(
                    brightness=aug_config.brightness,
                    contrast=aug_config.contrast,
                    saturation=aug_config.saturation,
                    hue=aug_config.hue
                )
            else:
                print(f"지원하지 않는 증강 기법: {aug_type}")
                continue
            
            augmentations.append(aug)
        
        # 첫 번째 모델로 TTA 수행
        if model_configs:
            first_model_config = model_configs[0]
            model = first_model_config['model_class'](**first_model_config['model_params'])
            
            # 체크포인트 로드
            checkpoint = torch.load(first_model_config['checkpoint_path'], map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # TTA 예측
            tta_filenames, tta_predictions = predict_with_tta(
                model=model,
                test_loader=test_loader,
                augmentations=augmentations,
                device=device
            )
            
            # TTA 결과 저장
            tta_predictions_path = experiment_dir / "tta_predictions.csv"
            tta_config = {
                'method': 'tta',
                'num_augmentations': len(augmentations),
                'augmentation_types': [aug_config.type for aug_config in cfg.ensemble.tta.augmentations]
            }
            
            save_ensemble_predictions(
                filenames=tta_filenames,
                predictions=tta_predictions,
                ensemble_config=tta_config,
                output_path=str(tta_predictions_path),
                class_names=class_names,
                use_sample_order=True
            )
    
    # wandb에 결과 로깅
    wandb.log({
        'ensemble_method': cfg.ensemble.method,
        'num_models': len(cfg.ensemble.models),
        'model_names': [model.name for model in cfg.ensemble.models],
        'weights': weights if cfg.ensemble.method == 'weighted' else None
    })
    
    print("=== 앙상블 실험 완료 ===")
    print(f"앙상블 예측 결과: {predictions_path}")
    
    # wandb 종료
    wandb.finish()


if __name__ == "__main__":
    main() 