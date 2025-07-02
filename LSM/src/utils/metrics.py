import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import Dict, List, Tuple


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """평가 지표 계산"""
    
    # 텐서를 numpy 배열로 변환
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 예측값을 클래스로 변환
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1
    }


def get_classification_report(y_true: torch.Tensor, y_pred: torch.Tensor, target_names: List[str] = None) -> str:
    """분류 리포트 생성"""
    
    # 텐서를 numpy 배열로 변환
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 예측값을 클래스로 변환
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    return classification_report(y_true, y_pred, target_names=target_names) 