#!/usr/bin/env python3
"""
문서 이미지 분류 실험 실행 스크립트
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lightning_modules.classification_module import ClassificationModule
from src.utils.data_loader import create_data_loaders
from src.utils.prediction import predict_test_set, save_predictions_in_sample_order
from src.utils.s3_utils import create_s3_handler
from src.utils.env_utils import check_required_env_vars, get_wandb_config
from src.datasets.basic import BasicDataset
from src.datasets.test_dataset import TestDataset
from src.transforms.basic import BasicTransform


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """메인 함수"""
    
    print("=== 실험 시작 ===")
    print(f"실험 이름: {cfg.experiment.name}")
    print(f"설명: {cfg.experiment.description}")
    print(f"태그: {cfg.experiment.tags}")
    
    # 환경 변수 확인
    check_required_env_vars()
    
    # 시드 설정
    pl.seed_everything(cfg.seed)
    
    # 데이터 로더 생성
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    train_loader, val_loader = create_data_loaders(config_dict)
    
    # transform 생성 (테스트용)
    transform = BasicTransform(
        image_size=cfg.transform.image_size
    )
    
    # 테스트 데이터 로더 생성 (test 폴더의 이미지들을 직접 로드)
    test_dataset = TestDataset(
        img_dir='data/test',
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    # 클래스명 (임시로 0-16까지)
    class_names = [f"class_{i}" for i in range(cfg.num_classes)]
    
    # 모델 생성
    model = ClassificationModule(OmegaConf.to_container(cfg, resolve=True))
    
    # 실험 디렉터리 생성
    experiment_dir = Path(cfg.save.model_dir) / cfg.experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 콜백 설정
    callbacks = [
        ModelCheckpoint(
            monitor='val_f1',
            dirpath=experiment_dir,
            filename='best-{epoch:02d}-{val_f1:.4f}',
            save_top_k=cfg.logging.save_top_k,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_f1',
            patience=cfg.training.patience,
            mode='max'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # wandb 설정 가져오기
    wandb_config = get_wandb_config()
    
    # 로거 설정
    logger = WandbLogger(
        project=cfg.logging.project_name,
        name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # 트레이너 설정
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps
    )
    
    # 학습 시작
    trainer.fit(model, train_loader, val_loader)
    
    # 테스트 예측 수행
    print("테스트 예측 수행 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델을 device로 이동
    model = model.to(device)
    
    image_ids, predictions = predict_test_set(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # 예측 결과 저장
    predictions_path = experiment_dir / "predictions.csv"
    save_predictions_in_sample_order(
        filenames=image_ids,
        predictions=predictions,
        output_path=str(predictions_path)
    )
    
    # S3에 모델 저장 (활성화된 경우)
    if cfg.s3.enabled:
        s3_handler = create_s3_handler()
        if s3_handler:
            model_path = experiment_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            s3_handler.upload_model(str(model_path), cfg.experiment.name)
    
    print("=== 실험 완료 ===")
    print(f"예측 결과: {predictions_path}")
    
    # wandb 종료
    wandb.finish()


if __name__ == '__main__':
    main()