#!/usr/bin/env python3
"""
클래스별 메트릭 사용 예제 스크립트

이 스크립트는 학습된 모델을 사용하여 클래스별 정확도, F1 점수, 손실을 측정하고
시각화하는 방법을 보여줍니다.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lightning_modules.classification_module import ClassificationModule
from src.utils.data_loader import create_data_loaders
from src.utils.class_metrics import (
    evaluate_model_with_class_metrics,
    print_class_performance_summary,
    save_class_metrics_to_csv,
    plot_class_performance,
    plot_confusion_matrix,
    compare_train_val_performance
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """클래스별 메트릭 평가 메인 함수"""
    
    print("=== 클래스별 메트릭 평가 시작 ===")
    
    # 설정을 딕셔너리로 변환
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 클래스명 설정 (실제 클래스명으로 변경하세요)
    class_names = [
        "class_0", "class_1", "class_2", "class_3", "class_4",
        "class_5", "class_6", "class_7", "class_8", "class_9",
        "class_10", "class_11", "class_12", "class_13", "class_14",
        "class_15", "class_16"
    ]
    
    # 데이터 로더 생성
    train_loader, val_loader = create_data_loaders(config_dict)
    
    # 모델 로드 (체크포인트 경로를 지정하세요)
    checkpoint_path = "experiments/your_experiment_name/best-epoch=XX-val_f1=0.XXXX.ckpt"
    
    if os.path.exists(checkpoint_path):
        print(f"모델 로드 중: {checkpoint_path}")
        model = ClassificationModule.load_from_checkpoint(
            checkpoint_path,
            config=config_dict
        )
        model.set_class_names(class_names)
    else:
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("새로운 모델을 생성합니다.")
        model = ClassificationModule(config_dict)
        model.set_class_names(class_names)
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # 출력 디렉터리 생성
    output_dir = Path("class_metrics_output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n=== 검증 데이터로 클래스별 성능 평가 ===")
    
    # 검증 데이터로 클래스별 성능 평가
    val_results = evaluate_model_with_class_metrics(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names
    )
    
    # 클래스별 성능 요약 출력
    print_class_performance_summary(val_results, class_names)
    
    # 클래스별 메트릭을 CSV로 저장
    val_metrics_path = output_dir / "val_class_metrics.csv"
    save_class_metrics_to_csv(val_results, str(val_metrics_path), class_names)
    
    # 클래스별 성능 시각화
    val_plot_path = output_dir / "val_class_performance.png"
    plot_class_performance(val_results, str(val_plot_path), class_names)
    
    # 혼동 행렬 시각화
    confusion_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(val_results, str(confusion_plot_path), class_names)
    
    print("\n=== 훈련 데이터로 클래스별 성능 평가 ===")
    
    # 훈련 데이터로도 클래스별 성능 평가 (과적합 분석용)
    train_results = evaluate_model_with_class_metrics(
        model=model,
        dataloader=train_loader,
        device=device,
        class_names=class_names
    )
    
    # 훈련 vs 검증 성능 비교
    comparison_plot_path = output_dir / "train_val_comparison.png"
    compare_train_val_performance(
        train_results, val_results, str(comparison_plot_path), class_names
    )
    
    print("\n=== 결과 요약 ===")
    print(f"검증 클래스별 메트릭: {val_metrics_path}")
    print(f"검증 성능 그래프: {val_plot_path}")
    print(f"혼동 행렬: {confusion_plot_path}")
    print(f"훈련-검증 비교: {comparison_plot_path}")
    
    # 클래스별 성능 분석
    print("\n=== 클래스별 성능 분석 ===")
    
    if 'class_metrics' in val_results:
        # 가장 성능이 좋은 클래스
        best_class = max(val_results['class_metrics'].items(), 
                        key=lambda x: x[1]['accuracy'])
        print(f"가장 성능이 좋은 클래스: {best_class[0]} (정확도: {best_class[1]['accuracy']:.4f})")
        
        # 가장 성능이 나쁜 클래스
        worst_class = min(val_results['class_metrics'].items(), 
                         key=lambda x: x[1]['accuracy'])
        print(f"가장 성능이 나쁜 클래스: {worst_class[0]} (정확도: {worst_class[1]['accuracy']:.4f})")
        
        # 평균 정확도
        avg_accuracy = sum(metrics['accuracy'] for metrics in val_results['class_metrics'].values()) / len(val_results['class_metrics'])
        print(f"평균 정확도: {avg_accuracy:.4f}")
        
        # 정확도 표준편차
        accuracies = [metrics['accuracy'] for metrics in val_results['class_metrics'].values()]
        import numpy as np
        std_accuracy = np.std(accuracies)
        print(f"정확도 표준편차: {std_accuracy:.4f}")
        
        # 성능이 낮은 클래스들 (평균 - 표준편차 미만)
        low_performance_classes = [
            class_name for class_name, metrics in val_results['class_metrics'].items()
            if metrics['accuracy'] < avg_accuracy - std_accuracy
        ]
        if low_performance_classes:
            print(f"성능이 낮은 클래스들: {low_performance_classes}")
    
    print("\n=== 클래스별 메트릭 평가 완료 ===")


def evaluate_single_class_performance(model, dataloader, target_class, device='cuda'):
    """
    특정 클래스의 성능만 평가하는 함수
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        target_class: 평가할 클래스 인덱스
        device: 사용할 디바이스
    
    Returns:
        해당 클래스의 성능 메트릭
    """
    model.eval()
    correct = 0
    total = 0
    class_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, targets = batch
            else:
                continue
            
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            
            # 해당 클래스의 샘플만 필터링
            class_mask = (targets == target_class)
            if class_mask.sum() > 0:
                class_outputs = outputs[class_mask]
                class_targets = targets[class_mask]
                
                # 정확도 계산
                pred_classes = class_outputs.argmax(dim=1)
                correct += (pred_classes == class_targets).sum().item()
                total += class_targets.size(0)
                
                # 손실 계산
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
                losses = criterion(class_outputs, class_targets)
                class_losses.extend(losses.cpu().numpy())
    
    if total > 0:
        accuracy = correct / total
        avg_loss = sum(class_losses) / len(class_losses) if class_losses else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'sample_count': total
        }
    else:
        return {
            'accuracy': 0.0,
            'loss': 0.0,
            'sample_count': 0
        }


if __name__ == '__main__':
    main() 