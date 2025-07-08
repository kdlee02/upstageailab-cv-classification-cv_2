"""
클래스별 메트릭 계산을 위한 유틸리티 함수들
"""

import torch
import numpy as np
import pandas as pd
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


def calculate_class_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                          num_classes: int, class_names: Optional[List[str]] = None) -> Dict:
    """
    클래스별 메트릭을 계산합니다.
    
    Args:
        predictions: 모델 예측 (logits)
        targets: 타겟 라벨 (라벨 스무딩이 적용된 경우 자동으로 처리)
        num_classes: 클래스 수
        class_names: 클래스명 리스트 (선택사항)
    
    Returns:
        클래스별 메트릭 딕셔너리
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    # 라벨 스무딩이 적용된 targets를 클래스 인덱스로 변환
    if targets.dim() > 1:
        targets = targets.argmax(dim=1)
    
    # targets를 long 타입으로 변환
    targets = targets.long()
    
    # 예측 클래스
    pred_classes = predictions.argmax(dim=1)
    
    # 메트릭 계산
    accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='none')
    f1_score = F1Score(task='multiclass', num_classes=num_classes, average='none')
    confusion = ConfusionMatrix(task='multiclass', num_classes=num_classes)
    
    class_accuracies = accuracy(predictions, targets)
    class_f1_scores = f1_score(predictions, targets)
    confusion_matrix = confusion(predictions, targets)
    
    # 클래스별 손실 계산 (CrossEntropyLoss 사용)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    losses = criterion(predictions, targets)
    
    class_losses = {}
    class_sample_counts = {}
    
    for class_idx in range(num_classes):
        class_mask = (targets == class_idx)
        if class_mask.sum() > 0:
            class_losses[class_idx] = losses[class_mask].mean().item()
            class_sample_counts[class_idx] = class_mask.sum().item()
        else:
            class_losses[class_idx] = 0.0
            class_sample_counts[class_idx] = 0
    
    # 결과 요약
    results = {
        'class_metrics': {},
        'overall_metrics': {
            'accuracy': (pred_classes == targets).float().mean().item(),
            'f1_macro': f1_score(predictions, targets).mean().item(),
            'confusion_matrix': confusion_matrix.numpy()
        }
    }
    
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f'class_{i}'
        results['class_metrics'][class_name] = {
            'accuracy': class_accuracies[i].item(),
            'f1_score': class_f1_scores[i].item(),
            'loss': class_losses[i],
            'sample_count': class_sample_counts[i]
        }
    
    return results


def evaluate_model_with_class_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                                    device: str = 'cuda', class_names: Optional[List[str]] = None) -> Dict:
    """
    모델을 평가하고 클래스별 메트릭을 계산합니다.
    
    Args:
        model: 평가할 모델
        dataloader: 평가용 데이터 로더
        device: 사용할 디바이스
        class_names: 클래스명 리스트 (선택사항)
    
    Returns:
        평가 결과 딕셔너리
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, targets = batch
            else:
                images = batch
                targets = None
            
            images = images.to(device)
            outputs = model(images)
            
            if targets is not None:
                all_targets.append(targets.cpu())
            all_predictions.append(outputs.cpu())
    
    all_predictions = torch.cat(all_predictions)
    
    if all_targets:
        all_targets = torch.cat(all_targets)
        num_classes = all_predictions.shape[1]
        
        return calculate_class_metrics(
            predictions=all_predictions,
            targets=all_targets,
            num_classes=num_classes,
            class_names=class_names
        )
    else:
        return {
            'predictions': all_predictions.argmax(dim=1).numpy(),
            'outputs': all_predictions.numpy()
        }


def print_class_performance_summary(results: Dict, class_names: Optional[List[str]] = None):
    """
    클래스별 성능 요약을 출력합니다.
    
    Args:
        results: calculate_class_metrics의 결과
        class_names: 클래스명 리스트 (선택사항)
    """
    print("=" * 60)
    print("클래스별 성능 요약")
    print("=" * 60)
    
    if 'overall_metrics' in results:
        print(f"전체 정확도: {results['overall_metrics']['accuracy']:.4f}")
        print(f"전체 F1 점수 (Macro): {results['overall_metrics']['f1_macro']:.4f}")
        print()
    
    if 'class_metrics' in results:
        print(f"{'클래스':<15} {'정확도':<10} {'F1 점수':<10} {'손실':<10} {'샘플 수':<10}")
        print("-" * 60)
        
        for class_name, metrics in results['class_metrics'].items():
            print(f"{class_name:<15} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} "
                  f"{metrics['loss']:<10.4f} {metrics['sample_count']:<10}")


def save_class_metrics_to_csv(results: Dict, output_path: str, class_names: Optional[List[str]] = None):
    """
    클래스별 메트릭을 CSV 파일로 저장합니다.
    
    Args:
        results: calculate_class_metrics의 결과
        output_path: 저장할 파일 경로
        class_names: 클래스명 리스트 (선택사항)
    """
    if 'class_metrics' not in results:
        print("클래스 메트릭이 없습니다.")
        return
    
    # 데이터프레임 생성
    data = []
    for class_name, metrics in results['class_metrics'].items():
        data.append({
            'class_name': class_name,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'loss': metrics['loss'],
            'sample_count': metrics['sample_count']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"클래스별 메트릭이 {output_path}에 저장되었습니다.")


def plot_class_performance(results: Dict, output_path: Optional[str] = None, 
                         class_names: Optional[List[str]] = None):
    """
    클래스별 성능을 시각화합니다.
    
    Args:
        results: calculate_class_metrics의 결과
        output_path: 저장할 파일 경로 (선택사항)
        class_names: 클래스명 리스트 (선택사항)
    """
    if 'class_metrics' not in results:
        print("클래스 메트릭이 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('클래스별 성능 분석', fontsize=16)
    
    # 데이터 준비
    classes = list(results['class_metrics'].keys())
    accuracies = [results['class_metrics'][c]['accuracy'] for c in classes]
    f1_scores = [results['class_metrics'][c]['f1_score'] for c in classes]
    losses = [results['class_metrics'][c]['loss'] for c in classes]
    sample_counts = [results['class_metrics'][c]['sample_count'] for c in classes]
    
    # 1. 정확도 막대 그래프
    axes[0, 0].bar(classes, accuracies, color='skyblue')
    axes[0, 0].set_title('클래스별 정확도')
    axes[0, 0].set_ylabel('정확도')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. F1 점수 막대 그래프
    axes[0, 1].bar(classes, f1_scores, color='lightgreen')
    axes[0, 1].set_title('클래스별 F1 점수')
    axes[0, 1].set_ylabel('F1 점수')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 손실 막대 그래프
    axes[1, 0].bar(classes, losses, color='salmon')
    axes[1, 0].set_title('클래스별 손실')
    axes[1, 0].set_ylabel('손실')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 샘플 수 막대 그래프
    axes[1, 1].bar(classes, sample_counts, color='gold')
    axes[1, 1].set_title('클래스별 샘플 수')
    axes[1, 1].set_ylabel('샘플 수')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"성능 그래프가 {output_path}에 저장되었습니다.")
    
    plt.show()


def plot_confusion_matrix(results: Dict, output_path: Optional[str] = None, 
                         class_names: Optional[List[str]] = None):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        results: calculate_class_metrics의 결과
        output_path: 저장할 파일 경로 (선택사항)
        class_names: 클래스명 리스트 (선택사항)
    """
    if 'overall_metrics' not in results or 'confusion_matrix' not in results['overall_metrics']:
        print("혼동 행렬이 없습니다.")
        return
    
    confusion_matrix = results['overall_metrics']['confusion_matrix']
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(confusion_matrix))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('혼동 행렬')
    plt.xlabel('예측 클래스')
    plt.ylabel('실제 클래스')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"혼동 행렬이 {output_path}에 저장되었습니다.")
    
    plt.show()


def compare_train_val_performance(train_results: Dict, val_results: Dict, 
                                output_path: Optional[str] = None,
                                class_names: Optional[List[str]] = None):
    """
    훈련과 검증 성능을 비교합니다.
    
    Args:
        train_results: 훈련 결과
        val_results: 검증 결과
        output_path: 저장할 파일 경로 (선택사항)
        class_names: 클래스명 리스트 (선택사항)
    """
    if 'class_metrics' not in train_results or 'class_metrics' not in val_results:
        print("훈련 또는 검증 메트릭이 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('훈련 vs 검증 성능 비교', fontsize=16)
    
    classes = list(train_results['class_metrics'].keys())
    
    # 정확도 비교
    train_acc = [train_results['class_metrics'][c]['accuracy'] for c in classes]
    val_acc = [val_results['class_metrics'][c]['accuracy'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_acc, width, label='훈련', color='skyblue')
    axes[0, 0].bar(x + width/2, val_acc, width, label='검증', color='lightcoral')
    axes[0, 0].set_title('클래스별 정확도 비교')
    axes[0, 0].set_ylabel('정확도')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45)
    axes[0, 0].legend()
    
    # F1 점수 비교
    train_f1 = [train_results['class_metrics'][c]['f1_score'] for c in classes]
    val_f1 = [val_results['class_metrics'][c]['f1_score'] for c in classes]
    
    axes[0, 1].bar(x - width/2, train_f1, width, label='훈련', color='lightgreen')
    axes[0, 1].bar(x + width/2, val_f1, width, label='검증', color='orange')
    axes[0, 1].set_title('클래스별 F1 점수 비교')
    axes[0, 1].set_ylabel('F1 점수')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes, rotation=45)
    axes[0, 1].legend()
    
    # 손실 비교
    train_loss = [train_results['class_metrics'][c]['loss'] for c in classes]
    val_loss = [val_results['class_metrics'][c]['loss'] for c in classes]
    
    axes[1, 0].bar(x - width/2, train_loss, width, label='훈련', color='gold')
    axes[1, 0].bar(x + width/2, val_loss, width, label='검증', color='lightpink')
    axes[1, 0].set_title('클래스별 손실 비교')
    axes[1, 0].set_ylabel('손실')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes, rotation=45)
    axes[1, 0].legend()
    
    # 과적합 분석 (훈련 - 검증 차이)
    acc_diff = [t - v for t, v in zip(train_acc, val_acc)]
    f1_diff = [t - v for t, v in zip(train_f1, val_f1)]
    
    axes[1, 1].bar(x, acc_diff, color='red', alpha=0.7, label='정확도 차이')
    axes[1, 1].bar(x, f1_diff, color='blue', alpha=0.7, label='F1 점수 차이')
    axes[1, 1].set_title('과적합 분석 (훈련 - 검증)')
    axes[1, 1].set_ylabel('차이')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"성능 비교 그래프가 {output_path}에 저장되었습니다.")
    
    plt.show() 