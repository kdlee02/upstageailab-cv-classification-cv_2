# 문서 이미지 분류 프로젝트

이 프로젝트는 문서 이미지를 분류하는 딥러닝 모델을 구현하고 다양한 앙상블 기법을 적용할 수 있는 프레임워크입니다.

## 주요 기능

- 다양한 모델 지원 (ResNet50, EfficientNet, Vision Transformer)
- 다양한 손실 함수 (Cross Entropy, Focal Loss)
- 다양한 옵티마이저 (Adam, AdamW, SGD)
- **다양한 스캐줄러 지원 (Cosine, Step, Exponential, Plateau, Warmup Cosine, Warmup Linear, Cosine Warm Restart)**
- 데이터 증강 기법
- **앙상블 기법 지원**
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