# Document Type Classification - Computer Vision Project

## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이승민](https://github.com/UpstageAILab)             |            [최웅비](https://github.com/UpstageAILab)             |            [이경도](https://github.com/UpstageAILab)             |            [이상원](https://github.com/UpstageAILab)             |            [김재덕](https://github.com/UpstageAILab)             |
|                            팀장, model 실험                             |                            담당 오프라인 데이터 증강 실험                             |                            담당 온라인 데이터 증강 실험                           |                            담당 model 실험                             |                            담당 비지도 학습 실험                             |

## 0. Overview
### Environment
- Python 3.10
- PyTorch, TorchLightning
- Albumentations, Augraphy
- WandB, Hydra
- VSCode, Cursor

### Requirements
- 이미지 분류 문제 해결
- 도메인 갭 해소를 위한 증강 기법 탐색
- 고성능 모델 설계 및 학습
- 잘못된 라벨 식별 및 정정

## 1. Competiton Info

### Overview

- 문서 타입 분류 대회 (Document Type Classification)
- 실제 환경에서 촬영된 다양한 문서 이미지에 대해 클래스 분류 모델 개발
- 주요 과제: 도메인 갭 문제 해결, 소수 클래스 처리, 하이퍼파라미터 최적화

### Timeline

- 2025.06.30 - 대회 시작
- 2025.07.10 - 제출 마감

## 2. Project Structure

### JL Directory Structure

```
JL/
├── configs/                    # Hydra configuration files
│   ├── callback/              # Callback configurations
│   │   └── early_stopping.yaml
│   ├── data/                  # Data module configurations
│   │   ├── convnextArcFace.yaml
│   │   ├── efficientnet.yaml
│   │   └── swinTransformer.yaml
│   ├── loss/                  # Loss function configurations
│   │   ├── crossentropyloss.yaml
│   │   └── focalloss.yaml
│   ├── model/                 # Model configurations
│   │   ├── convnextArcFace.yaml
│   │   ├── efficientnet.yaml
│   │   └── swinTransformer.yaml
│   ├── optimizer/             # Optimizer configurations
│   │   ├── adam.yaml
│   │   ├── adamw.yaml
│   │   └── sgd.yaml
│   ├── scheduler/             # Learning rate scheduler configurations
│   │   ├── cosineAnnealing_lr.yaml
│   │   └── step_lr.yaml
│   ├── trainer/               # Trainer configurations
│   │   └── trainer.yaml
│   └── config.yaml           # Main configuration file
├── datasets/                  # Dataset files
│   ├── meta.csv
│   ├── sample_submission.csv
│   └── train.csv
├── script/                    # Main execution scripts
│   ├── gradcam.py            # Grad-CAM visualization
│   ├── test.py               # Model testing
│   ├── train.py              # Model training
│   └── tta.py                # Test-time augmentation
├── src/                       # Source code modules
│   ├── callbacks/            # Custom callbacks
│   │   ├── averaging.py
│   │   ├── confusionMatrix.py
│   │   ├── evaluation.py
│   │   ├── hardNegativeMining.py
│   │   └── umap.py
│   ├── datasets/             # Dataset modules
│   ├── losses/               # Custom loss functions
│   ├── models/               # Model implementations
│   ├── trainer/              # Training modules
│   └── utils/                # Utility functions
├── .env.template             # Environment variables template
├── EDA.ipynb                 # Exploratory Data Analysis notebook
└── organize_images.py        # Image organization script
```

### Key Components

- **Hydra Configuration System**: Modular configuration management for experiments
- **PyTorch Lightning**: Training framework with callbacks and logging
- **TIMM Models**: Pre-trained vision models (EfficientNet, ConvNeXt, Swin Transformer)
- **Custom Callbacks**: Confusion matrix, UMAP visualization, hard negative mining
- **Flexible Loss Functions**: Cross-entropy, Focal loss with class balancing

## 3. Data descrption

### Dataset overview
- ![](../1.png)
- ![](../2.png)
- 클래스별 이미지 수 불균형 존재
- train/test 데이터 간 분포 차이 극심 (회전각 평균: train 1.9°, test 12.6°)
- 테스트셋은 실제 환경 기반의 비스듬한 촬영, 그림자, 조명 간섭 등 존재

### EDA
- ![](../3.png)
- 회전 각도, 밝기, 노이즈 등에서 정량/정성 차이 분석
- Grad-CAM, Confusion Matrix, FAISS+CLIP 기반 군집 시각화 활용

### Data Processing
- 증강 전략: 회전, 반전, 밝기/대비 조정, GaussianBlur, Normalize 등
- 오버샘플링 및 Mixup 적용
- 잘못 라벨링 데이터 정정 (CLIP → FAISS → FiftyOne 수동 검토)  

## 4. Modeling

### Model Description
- ResNet50, EfficientNet, ConvNeXt 등 모델 비교 실험
- 최종 채택: ConvNeXt + Custom Loss
- Loss Function: CB-Focal Loss (Focal + Class-Balanced Loss)
- Optimizer: AdamW + CosineAnnealing

### Modeling Process

- 증강: Augraphy(사전), Albumentations(학습 중) → 성능 향상 (F1 0.9363 → 0.9577)
- Hard Negative Mining & OHEM 실험 (효과 제한적)
- Test-Time Augmentation(TTA) 적용 시 오히려 성능 하락 (F1 0.9598 → 0.9553)
- 학습 중 Confusion Matrix 시각화 / 학습 후 Grad-CAM 사용
- ![](../4.png)
- ![](../5.png)
- ![](../6.png)

## 5. Getting Started (JL Implementation)

### Setup Environment

1. **Clone the repository and navigate to JL directory**
```bash
cd JL
```

2. **Create environment file**
```bash
cp .env.template .env
# Add your WANDB_API_KEY to .env file
```

3. **Install dependencies**
```bash
pip install torch torchvision pytorch-lightning
pip install timm albumentations hydra-core wandb
pip install scikit-learn matplotlib seaborn
```

### Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration file
- `configs/model/`: Model configurations (EfficientNet, ConvNeXt, Swin Transformer)
- `configs/data/`: Data module configurations
- `configs/optimizer/`: Optimizer settings (Adam, AdamW, SGD)
- `configs/scheduler/`: Learning rate schedulers
- `configs/loss/`: Loss function configurations

### Training

```bash
# Basic training with default EfficientNet configuration
python script/train.py

# Train with specific model
python script/train.py model=convnextArcFace

# Train with custom experiment name
python script/train.py experiment_name=my_experiment

# Override specific parameters
python script/train.py model.num_classes=17 optimizer.lr=0.001
```

### Testing and Inference

```bash
# Run inference on test set
python script/test.py

# Generate Grad-CAM visualizations
python script/gradcam.py

# Apply Test-Time Augmentation
python script/tta.py
```

### Key Features

- **Modular Configuration**: Easy experiment management with Hydra
- **Multiple Model Support**: EfficientNet, ConvNeXt, Swin Transformer with ArcFace option
- **Advanced Loss Functions**: Focal Loss with Class-Balanced weighting
- **Rich Callbacks**: Confusion Matrix, UMAP visualization, Hard Negative Mining
- **Comprehensive Logging**: WandB integration for experiment tracking
- **Visualization Tools**: Grad-CAM, confusion matrices, UMAP embeddings

### Dataset Structure

Place your dataset in the following structure:
```
JL/datasets/
├── train.csv          # Training labels
├── meta.csv           # Metadata
├── sample_submission.csv
└── images/            # Image files (create this directory)
    ├── train/
    └── test/
```

## etc


### Reference
- https://github.com/facebookresearch/faiss
- https://albumentations.ai/
- https://augraphy.com/
- https://wandb.ai/
- https://pytorch.org/