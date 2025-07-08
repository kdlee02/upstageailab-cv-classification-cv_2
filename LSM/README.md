# 문서 이미지 분류 프로젝트

이 프로젝트는 문서 이미지를 분류하는 딥러닝 모델을 구현하고 다양한 앙상블 기법을 적용할 수 있는 프레임워크입니다.

## 주요 기능

- 다양한 모델 지원 (ResNet50, EfficientNet, Vision Transformer, **ConvNeXt**)
- 다양한 손실 함수 (Cross Entropy, Focal Loss)
- 다양한 옵티마이저 (Adam, AdamW, SGD)
- **다양한 스캐줄러 지원 (Cosine, Step, Exponential, Plateau, Warmup Cosine, Warmup Linear, Cosine Warm Restart)**
- 데이터 증강 기법
- **앙상블 기법 지원**
- **클래스별 성능 분석 및 시각화**
- 실험 관리 및 로깅 (Weights & Biases)
- **S3 모델 저장 지원**

## 앙상블 기법

이 프로젝트는 다음과 같은 앙상블 기법들을 지원합니다:

### 1. 모델 앙상블
- **투표 앙상블 (Voting)**: 각 모델의 예측 클래스에 대해 다수결 투표
- **평균 앙상블 (Averaging)**: 각 모델의 예측 확률을 평균
- **가중 앙상블 (Weighted)**: 각 모델에 가중치를 적용한 가중 평균
- **스태킹 앙상블 (Stacking)**: 메타 모델을 사용한 앙상블

### 2. 테스트 타임 증강 (TTA)
- 수평 뒤집기, 회전, 색상 변화 등의 증강 기법을 테스트 시에 적용
- 원본 이미지와 증강된 이미지들의 예측을 평균

### 3. 교차 검증 앙상블
- K-fold 교차 검증으로 훈련된 모델들의 앙상블

### 4. 스냅샷 앙상블
- 동일 모델의 서로 다른 체크포인트들을 앙상블

## 설치 및 설정

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
```bash
# 환경 변수 파일 복사
cp env_example.txt .env

# .env 파일 편집
nano .env  # 또는 원하는 텍스트 에디터 사용
```

### 환경 변수 설정

`.env` 파일에 다음 설정들을 추가하세요:

```bash
# Weights & Biases 설정 (필수)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=document-classification

# AWS S3 설정 (선택사항)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket-name

# 실험 설정
CUDA_VISIBLE_DEVICES=0
```

#### 필수 환경 변수
- `WANDB_API_KEY`: Weights & Biases API 키
- `WANDB_PROJECT`: 프로젝트 이름

#### 선택사항 환경 변수
- `S3_BUCKET_NAME`: S3 버킷 이름 (모델 저장용)
- `AWS_ACCESS_KEY_ID`: AWS 액세스 키
- `AWS_SECRET_ACCESS_KEY`: AWS 시크릿 키
- `AWS_DEFAULT_REGION`: AWS 리전 (기본값: us-east-1)
- `CUDA_VISIBLE_DEVICES`: 사용할 GPU 번호

## 사용 방법

### 기본 실험 실행

```bash
python run_experiment.py
```

### 앙상블 실험 실행

```bash
python run_ensemble_experiment.py
```

### 환경 변수 확인

```bash
python example_env_usage.py
```

### 스캐줄러 사용 예제

```bash
python example_scheduler_usage.py
```

### 클래스별 메트릭 사용 예제

```bash
python example_class_metrics_usage.py
```

### ConvNeXt 모델 사용 예제

```bash
python example_convnext_usage.py
```

### 스캐줄러 설정

다양한 스캐줄러를 사용할 수 있습니다. `configs/config.yaml`에서 기본 스캐줄러를 설정하거나, 명령줄에서 지정할 수 있습니다:

#### 기본 설정에서 스캐줄러 변경
```yaml
# configs/config.yaml
scheduler:
  name: "cosine"  # cosine, step, exponential, plateau, warmup_cosine, warmup_linear, cosine_warm_restart
```

#### 스캐줄러 파라미터 직접 설정
```yaml
# configs/config.yaml
scheduler:
  name: "cosine_warm_restart"
  params:
    T_0: 10
    T_mult: 2
    eta_min: 0.000001
```

#### 명령줄에서 스캐줄러 지정
```bash
# Cosine 스캐줄러 사용
python run_experiment.py scheduler=cosine

# Step 스캐줄러 사용
python run_experiment.py scheduler=step

# Warmup Cosine 스캐줄러 사용
python run_experiment.py scheduler=warmup_cosine

# Cosine Warm Restart 스캐줄러 사용
python run_experiment.py scheduler=cosine_warm_restart
```

#### 스캐줄러별 설정 예시

**Cosine 스캐줄러** (`configs/scheduler/cosine.yaml`):
```yaml
_target_: src.schedulers.cosine.CosineScheduler
T_max: 100
eta_min: 0.0
```

**Step 스캐줄러** (`configs/scheduler/step.yaml`):
```yaml
_target_: src.schedulers.step.StepScheduler
step_size: 30
gamma: 0.1
```

**Warmup Cosine 스캐줄러** (`configs/scheduler/warmup_cosine.yaml`):
```yaml
_target_: src.schedulers.warmup_cosine.WarmupCosineScheduler
warmup_steps: 1000
max_steps: 10000
min_lr: 0.0
warmup_start_lr: 0.0
```

**Plateau 스캐줄러** (`configs/scheduler/plateau.yaml`):
```yaml
_target_: src.schedulers.plateau.PlateauScheduler
mode: min
factor: 0.1
patience: 10
verbose: false
```

**Cosine Warm Restart 스캐줄러** (`configs/scheduler/cosine_warm_restart.yaml`):
```yaml
_target_: src.schedulers.cosine_warm_restart.CosineWarmRestartScheduler
T_0: 10
T_mult: 2
eta_min: 0.000001
```

## 클래스별 성능 분석

이 프로젝트는 클래스별 정확도, F1 점수, 손실을 측정하고 시각화하는 기능을 제공합니다.

### 주요 기능

- **클래스별 정확도**: 각 클래스별 개별 정확도 측정
- **클래스별 F1 점수**: 각 클래스별 F1 점수 측정
- **클래스별 손실**: 각 클래스별 평균 손실 계산
- **혼동 행렬**: 클래스 간 예측 오류 분석
- **훈련-검증 비교**: 과적합 분석을 위한 훈련/검증 성능 비교
- **시각화**: 클래스별 성능을 그래프로 시각화

### 사용 방법

#### 1. 기본 실험에서 클래스별 메트릭 자동 생성

`run_experiment.py`를 실행하면 자동으로 다음 파일들이 생성됩니다:

- `val_class_metrics.csv`: 검증 데이터의 클래스별 메트릭
- `val_class_performance.png`: 클래스별 성능 그래프
- `confusion_matrix.png`: 혼동 행렬
- `train_val_comparison.png`: 훈련-검증 성능 비교

#### 2. 별도 클래스별 메트릭 평가

```bash
python example_class_metrics_usage.py
```

#### 3. 특정 클래스 성능만 평가

```python
from src.utils.class_metrics import evaluate_single_class_performance

# 특정 클래스(예: 클래스 0)의 성능만 평가
class_0_performance = evaluate_single_class_performance(
    model=model,
    dataloader=val_loader,
    target_class=0,
    device='cuda'
)
print(f"클래스 0 정확도: {class_0_performance['accuracy']:.4f}")
```

### 출력 예시

```
============================================================
클래스별 성능 요약
============================================================
전체 정확도: 0.8542
전체 F1 점수 (Macro): 0.8234

클래스          정확도     F1 점수   손실      샘플 수  
------------------------------------------------------------
class_0        0.9234     0.9123     0.1234    150      
class_1        0.8765     0.8543     0.2345    120      
class_2        0.7890     0.7654     0.3456    180      
...
```

### 시각화 예시

- **클래스별 정확도 막대 그래프**: 각 클래스의 정확도를 한눈에 비교
- **클래스별 F1 점수 막대 그래프**: 각 클래스의 F1 점수 비교
- **클래스별 손실 막대 그래프**: 각 클래스의 평균 손실 비교
- **혼동 행렬 히트맵**: 클래스 간 예측 오류 패턴 분석
- **훈련-검증 비교 그래프**: 과적합 여부 분석

## ConvNeXt 모델

ConvNeXt는 Vision Transformer의 설계 원칙을 CNN에 적용한 최신 모델입니다.

### 지원하는 ConvNeXt 모델

- **ConvNeXt Tiny**: 가장 작은 모델 (28M 파라미터)
- **ConvNeXt Small**: 작은 모델 (50M 파라미터)
- **ConvNeXt Base**: 중간 크기 모델 (88M 파라미터)
- **ConvNeXt Large**: 큰 모델 (198M 파라미터)

### ConvNeXt 변형 모델

- **기본 ConvNeXt**: 표준 ConvNeXt 모델
- **ConvNeXt + Attention**: 어텐션 메커니즘 추가
- **ConvNeXt + Focal Loss**: Focal Loss 최적화

### 사용 방법

#### 1. 명령줄에서 ConvNeXt 모델 지정

```bash
# ConvNeXt Tiny 사용
python run_experiment.py model=convnext

# ConvNeXt Small 사용
python run_experiment.py model=convnext_small

# ConvNeXt Base 사용
python run_experiment.py model=convnext_base

# ConvNeXt Large 사용
python run_experiment.py model=convnext_large

# 어텐션 메커니즘 추가
python run_experiment.py model=convnext_with_attention

# Focal Loss 최적화
python run_experiment.py model=convnext_with_focal
```

#### 2. 설정 파일에서 ConvNeXt 모델 설정

```yaml
# configs/config.yaml
model:
  name: convnext
  model_name: convnext_tiny  # convnext_tiny, convnext_small, convnext_base, convnext_large
  dropout_rate: 0.1
  use_attention: false
  use_focal_loss: false
  pretrained: true
  alpha: 1.0  # Focal Loss alpha (use_focal_loss가 true일 때)
  gamma: 2.0  # Focal Loss gamma (use_focal_loss가 true일 때)
  attention_dim: 256  # 어텐션 차원 (use_attention이 true일 때)
```

#### 3. 프로그래밍 방식으로 ConvNeXt 모델 생성

```python
from src.models.convnext import create_convnext_model

# 기본 ConvNeXt Tiny
model = create_convnext_model(
    model_name='convnext_tiny',
    num_classes=17,
    dropout_rate=0.1,
    pretrained=True
)

# 어텐션 메커니즘 추가
model = create_convnext_model(
    model_name='convnext_tiny',
    num_classes=17,
    dropout_rate=0.1,
    use_attention=True,
    attention_dim=256,
    pretrained=True
)

# Focal Loss 최적화
model = create_convnext_model(
    model_name='convnext_tiny',
    num_classes=17,
    dropout_rate=0.1,
    use_focal_loss=True,
    alpha=1.0,
    gamma=2.0,
    pretrained=True
)
```

### ConvNeXt 모델 특징

- **효율적인 아키텍처**: Vision Transformer의 설계 원칙을 CNN에 적용
- **높은 성능**: ImageNet에서 우수한 성능 달성
- **다양한 크기**: Tiny부터 Large까지 다양한 모델 크기 제공
- **확장 가능**: 어텐션 메커니즘, Focal Loss 등 다양한 변형 지원

### 앙상블 설정 예시

`configs/ensemble.yaml` 파일에서 앙상블 설정을 조정할 수 있습니다:

```yaml
ensemble:
  enabled: true
  method: "averaging"  # voting, averaging, weighted, stacking
  models:
    - name: "resnet50"
      checkpoint_path: "checkpoints/resnet50_best.ckpt"
      weight: 1.0
    - name: "efficientnet"
      checkpoint_path: "checkpoints/efficientnet_best.ckpt"
      weight: 1.0
    - name: "vit"
      checkpoint_path: "checkpoints/vit_best.ckpt"
      weight: 1.0
  
  # 테스트 타임 증강 설정
  tta:
    enabled: true
    augmentations:
      - type: "horizontal_flip"
        probability: 0.5
      - type: "rotation"
        degrees: [90, 180, 270]
```

### 다양한 앙상블 방법 사용

#### 1. 투표 앙상블
```yaml
ensemble:
  method: "voting"
  models:
    - name: "resnet50"
      checkpoint_path: "checkpoints/resnet50.ckpt"
    - name: "efficientnet"
      checkpoint_path: "checkpoints/efficientnet.ckpt"
```

#### 2. 가중 앙상블
```yaml
ensemble:
  method: "weighted"
  models:
    - name: "resnet50"
      checkpoint_path: "checkpoints/resnet50.ckpt"
      weight: 0.4
    - name: "efficientnet"
      checkpoint_path: "checkpoints/efficientnet.ckpt"
      weight: 0.6
```

#### 3. TTA 앙상블
```yaml
ensemble:
  tta:
    enabled: true
    augmentations:
      - type: "horizontal_flip"
        probability: 0.5
      - type: "color_jitter"
        brightness: 0.1
        contrast: 0.1
```

## 프로젝트 구조

```
base_code2/
├── configs/                 # 설정 파일들
│   ├── config.yaml         # 메인 설정
│   ├── ensemble.yaml       # 앙상블 설정
│   ├── model/              # 모델 설정
│   ├── optimizer/          # 옵티마이저 설정
│   ├── scheduler/          # 스캐줄러 설정
│   └── ...
├── src/
│   ├── models/             # 모델 구현
│   ├── optimizers/         # 옵티마이저 구현
│   ├── schedulers/         # 스캐줄러 구현
│   ├── utils/
│   │   ├── ensemble.py     # 앙상블 구현
│   │   ├── env_utils.py    # 환경 변수 유틸리티
│   │   ├── s3_utils.py     # S3 유틸리티
│   │   ├── prediction.py   # 예측 함수
│   │   └── ...
│   └── ...
├── run_experiment.py       # 기본 실험 실행
├── run_ensemble_experiment.py  # 앙상블 실험 실행
├── example_env_usage.py    # 환경 변수 사용 예시
├── example_ensemble_usage.py  # 앙상블 사용 예시
├── .env                    # 환경 변수 (생성 필요)
├── env_example.txt         # 환경 변수 예시
└── ...
```

## S3 모델 저장

S3를 사용하여 모델을 저장하려면:

1. `.env` 파일에 S3 설정 추가:
```bash
S3_BUCKET_NAME=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

2. 실험 실행 시 자동으로 S3에 모델이 저장됩니다.

## 앙상블 성능 향상 팁

1. **다양한 모델 사용**: 서로 다른 아키텍처의 모델을 조합하면 더 좋은 성능을 얻을 수 있습니다.

2. **가중치 조정**: 검증 성능에 따라 각 모델의 가중치를 조정하세요.

3. **TTA 활용**: 테스트 타임 증강은 특히 이미지 분류에서 효과적입니다.

4. **체크포인트 선택**: 최고 성능 체크포인트뿐만 아니라 다양한 에포크의 체크포인트도 고려해보세요.

## 결과 분석

앙상블 실험 결과는 다음 파일들로 저장됩니다:
- `ensemble_predictions.csv`: 앙상블 예측 결과
- `ensemble_predictions_meta.json`: 앙상블 메타데이터
- `tta_predictions.csv`: TTA 예측 결과 (활성화된 경우)

## 문제 해결

### 환경 변수 관련 문제
- `.env` 파일이 프로젝트 루트에 있는지 확인
- 필수 환경 변수가 설정되어 있는지 확인: `python example_env_usage.py`

### S3 관련 문제
- S3 버킷 이름이 올바르게 설정되어 있는지 확인
- AWS 자격 증명이 올바른지 확인
- S3 버킷에 대한 권한이 있는지 확인

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 