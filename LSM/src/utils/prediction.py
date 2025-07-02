import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import pandas as pd
from pathlib import Path
import numpy as np
import os

from src.utils.ensemble import (
    EnsembleModel, 
    TestTimeAugmentation, 
    create_ensemble_from_checkpoints
)


def predict_test_set(
    model, 
    test_loader: DataLoader, 
    device: str = 'cuda',
    ensemble_model: Optional[EnsembleModel] = None,
    tta_model: Optional[TestTimeAugmentation] = None
) -> Tuple[List[str], List[int]]:
    """
    테스트 세트에 대한 예측 수행
    
    Args:
        model: 단일 모델 또는 앙상블 모델
        test_loader: 테스트 데이터 로더
        device: 사용할 디바이스
        ensemble_model: 앙상블 모델 (선택사항)
        tta_model: TTA 모델 (선택사항)
    
    Returns:
        image_ids: 이미지 ID 리스트
        predictions: 예측 클래스 리스트
    """
    model.eval()
    # 모델을 device로 이동
    model = model.to(device)
    
    image_ids = []
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                images, batch_image_ids = batch
            else:
                images = batch
                batch_image_ids = [f"test_{i}" for i in range(len(images))]
            
            images = images.to(device)
            
            # 앙상블 모델 사용
            if ensemble_model is not None:
                outputs = ensemble_model.predict(images, device)
            # TTA 모델 사용
            elif tta_model is not None:
                outputs = tta_model.predict(images, device)
            # 단일 모델 사용
            else:
                outputs = model(images)
            
            # 소프트맥스 적용
            if not isinstance(outputs, torch.Tensor):
                outputs = torch.tensor(outputs)
            
            probs = F.softmax(outputs, dim=1)
            pred_classes = torch.argmax(probs, dim=1)
            
            image_ids.extend(batch_image_ids)
            predictions.extend(pred_classes.cpu().numpy().tolist())
    
    return image_ids, predictions


def predict_with_ensemble(
    model_configs: List[dict],
    test_loader: DataLoader,
    device: str = 'cuda',
    ensemble_method: str = 'averaging',
    weights: Optional[List[float]] = None
) -> Tuple[List[str], List[int]]:
    """
    여러 모델의 체크포인트를 사용한 앙상블 예측
    
    Args:
        model_configs: 모델 설정 리스트 [{'model_class': ..., 'model_params': ..., 'checkpoint_path': ...}]
        test_loader: 테스트 데이터 로더
        device: 사용할 디바이스
        ensemble_method: 앙상블 방법
        weights: 가중치 (가중 앙상블용)
    
    Returns:
        filenames: 파일명 리스트
        predictions: 예측 클래스 리스트
    """
    # 앙상블 모델 생성
    models = []
    for config in model_configs:
        model_class = config['model_class']
        model_params = config['model_params']
        checkpoint_path = config['checkpoint_path']
        
        # 모델 생성
        model = model_class(**model_params)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        models.append(model)
    
    # 앙상블 모델 생성
    ensemble_model = EnsembleModel(models, ensemble_method)
    
    # 가중치 설정
    if weights is not None:
        ensemble_model.set_weights(weights)
    
    # 예측 수행
    return predict_test_set(
        model=None, 
        test_loader=test_loader, 
        device=device, 
        ensemble_model=ensemble_model
    )


def predict_with_tta(
    model,
    test_loader: DataLoader,
    augmentations: List,
    device: str = 'cuda'
) -> Tuple[List[str], List[int]]:
    """
    테스트 타임 증강을 사용한 예측
    
    Args:
        model: 예측할 모델
        test_loader: 테스트 데이터 로더
        augmentations: 증강 기법 리스트
        device: 사용할 디바이스
    
    Returns:
        filenames: 파일명 리스트
        predictions: 예측 클래스 리스트
    """
    tta_model = TestTimeAugmentation(model, augmentations)
    
    return predict_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        tta_model=tta_model
    )


def save_predictions_to_csv(
    filenames: List[str], 
    predictions: List[int], 
    output_path: str,
    class_names: Optional[List[str]] = None
):
    """예측 결과를 CSV 파일로 저장 (ID, target 형식)"""
    df = pd.DataFrame({
        'ID': filenames,
        'target': predictions
    })
    
    df.to_csv(output_path, index=False)
    print(f"예측 결과가 {output_path}에 저장되었습니다.")
    print(f"총 {len(df)}개의 예측 결과가 저장되었습니다.")


def save_predictions_in_sample_order(
    filenames: List[str], 
    predictions: List[int], 
    output_path: str,
    sample_submission_path: str = "data/sample_submission.csv",
    class_names: Optional[List[str]] = None
):
    """
    sample_submission.csv의 ID 순서대로 예측 결과를 CSV 파일로 저장
    
    Args:
        filenames: 예측된 파일명 리스트
        predictions: 예측 클래스 리스트
        output_path: 저장할 CSV 파일 경로
        sample_submission_path: sample_submission.csv 파일 경로
        class_names: 클래스명 리스트 (선택사항)
    """
    # sample_submission.csv 읽기
    sample_df = pd.read_csv(sample_submission_path)
    
    # 예측 결과를 딕셔너리로 변환
    pred_dict = dict(zip(filenames, predictions))
    
    # sample_submission의 ID 순서대로 예측 결과 매핑
    ordered_predictions = []
    missing_ids = []
    
    for id_value in sample_df['ID']:
        ordered_predictions.append(pred_dict[id_value])
    
    # 결과 DataFrame 생성
    result_df = pd.DataFrame({
        'ID': sample_df['ID'],
        'target': ordered_predictions
    })
    
    # CSV 파일로 저장
    result_df.to_csv(output_path, index=False)
    
    print(f"예측 결과가 {output_path}에 저장되었습니다.")
    print(f"총 {len(result_df)}개의 예측 결과가 저장되었습니다.")
    
    if missing_ids:
        print(f"경고: {len(missing_ids)}개의 ID가 예측 결과에 없어 기본값 0으로 설정되었습니다.")
        print(f"첫 5개 누락 ID: {missing_ids[:5]}")
    
    # 예측 결과 검증
    if len(result_df) != len(sample_df):
        print(f"경고: 결과 행 수({len(result_df)})가 sample_submission 행 수({len(sample_df)})와 다릅니다.")
    


def save_ensemble_predictions(
    filenames: List[str],
    predictions: List[int],
    ensemble_config: dict,
    output_path: str,
    class_names: Optional[List[str]] = None,
    use_sample_order: bool = False,
    sample_submission_path: str = "data/sample_submission.csv"
):
    """
    앙상블 예측 결과를 CSV 파일로 저장 (메타데이터 포함)
    
    Args:
        filenames: 파일명 리스트
        predictions: 예측 클래스 리스트
        ensemble_config: 앙상블 설정
        output_path: 저장할 CSV 파일 경로
        class_names: 클래스명 리스트 (선택사항)
        use_sample_order: sample_submission.csv 순서 사용 여부
        sample_submission_path: sample_submission.csv 파일 경로
    """
    if use_sample_order:
        # sample_submission 순서 사용
        result_df = save_predictions_in_sample_order(
            filenames=filenames,
            predictions=predictions,
            output_path=output_path,
            sample_submission_path=sample_submission_path,
            class_names=class_names
        )
    else:
        # 기존 방식 사용
        df = pd.DataFrame({
            'filename': filenames,
            'prediction': predictions
        })
        
        # 클래스명이 제공된 경우 추가
        if class_names is not None:
            df['class_name'] = [class_names[pred] for pred in predictions]
        
        # 앙상블 메타데이터 추가
        df.attrs['ensemble_method'] = ensemble_config.get('method', 'unknown')
        df.attrs['num_models'] = ensemble_config.get('num_models', 0)
        df.attrs['weights'] = ensemble_config.get('weights', None)
        
        df.to_csv(output_path, index=False)
        result_df = df
    
    # 메타데이터를 별도 파일로 저장
    meta_path = output_path.replace('.csv', '_meta.json')
    import json
    with open(meta_path, 'w') as f:
        json.dump({
            'ensemble_method': ensemble_config.get('method', 'unknown'),
            'num_models': ensemble_config.get('num_models', 0),
            'weights': ensemble_config.get('weights', None),
            'model_configs': ensemble_config.get('models', []),
            'use_sample_order': use_sample_order
        }, f, indent=2)
    
    print(f"앙상블 예측 결과가 {output_path}에 저장되었습니다.")
    print(f"메타데이터가 {meta_path}에 저장되었습니다.")
    
    return result_df 