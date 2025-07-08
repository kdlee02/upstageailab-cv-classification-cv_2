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
from src.utils.class_metrics import (
    evaluate_model_with_class_metrics,
    print_class_performance_summary,
    save_class_metrics_to_csv,
    plot_class_performance,
    plot_confusion_matrix,
    compare_train_val_performance
)
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
    model.set_class_names(class_names)
    
    # 실험 디렉터리 생성
    experiment_dir = Path(cfg.save.model_dir) / cfg.experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=experiment_dir,
        filename='best-{epoch:02d}-{val_f1:.4f}',
        save_top_k=cfg.logging.save_top_k,
        mode='max'
    )
    
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor='val_f1',
            patience=cfg.training.patience,
            mode='max'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # wandb 설정 가져오기
    wandb_config = get_wandb_config()
    
    # wandb 파라미터가 설정되어 있는지 확인
    wandb_params = getattr(cfg, 'wandb', {})
    
    # 로거 설정
    logger = WandbLogger(
        project=wandb_params.get('project', cfg.logging.project_name),
        name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        entity=wandb_params.get('entity', None),
        group=wandb_params.get('group', None),
        job_type=wandb_params.get('job_type', 'training'),
        notes=wandb_params.get('notes', ''),
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # 트레이너 설정
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else 'auto',
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps
    )
    
    # 학습 시작
    trainer.fit(model, train_loader, val_loader)
    
    # 베스트 모델 로드
    print("베스트 모델 로드 중...")
    if checkpoint_callback and checkpoint_callback.best_model_path:
        best_model_path = checkpoint_callback.best_model_path
        print(f"베스트 모델 경로: {best_model_path}")
        # 베스트 모델 로드
        model = ClassificationModule.load_from_checkpoint(
            best_model_path,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        model.set_class_names(class_names)
    else:
        print("베스트 모델을 찾을 수 없습니다. 현재 모델을 사용합니다.")
    
    # 클래스별 성능 평가
    print("\n=== 클래스별 성능 평가 ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 검증 데이터로 클래스별 성능 평가
    print("검증 데이터로 클래스별 성능 평가 중...")
    val_results = evaluate_model_with_class_metrics(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names
    )
    
    # 클래스별 성능 요약 출력
    print_class_performance_summary(val_results, class_names)
    
    # 클래스별 메트릭을 CSV로 저장
    val_metrics_path = experiment_dir / "val_class_metrics.csv"
    save_class_metrics_to_csv(val_results, str(val_metrics_path), class_names)
    
    # 클래스별 성능 시각화
    val_plot_path = experiment_dir / "val_class_performance.png"
    plot_class_performance(val_results, str(val_plot_path), class_names)
    
    # 혼동 행렬 시각화
    confusion_plot_path = experiment_dir / "confusion_matrix.png"
    plot_confusion_matrix(val_results, str(confusion_plot_path), class_names)
    
    # 훈련 데이터로도 클래스별 성능 평가 (과적합 분석용)
    print("\n훈련 데이터로 클래스별 성능 평가 중...")
    train_results = evaluate_model_with_class_metrics(
        model=model,
        dataloader=train_loader,
        device=device,
        class_names=class_names
    )
    
    # 훈련 vs 검증 성능 비교
    comparison_plot_path = experiment_dir / "train_val_comparison.png"
    compare_train_val_performance(
        train_results, val_results, str(comparison_plot_path), class_names
    )
    
    # 테스트 예측 수행
    print("\n=== 테스트 예측 수행 중...")
    
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
            s3_path = f"models/{cfg.experiment.name}/best_model.pth"
            s3_handler.save_model(model, str(model_path), s3_path)
    
    print("=== 실험 완료 ===")
    print(f"예측 결과: {predictions_path}")
    print(f"검증 클래스별 메트릭: {val_metrics_path}")
    print(f"검증 성능 그래프: {val_plot_path}")
    print(f"혼동 행렬: {confusion_plot_path}")
    print(f"훈련-검증 비교: {comparison_plot_path}")
    
    # wandb 종료
    wandb.finish()


if __name__ == '__main__':
    main()