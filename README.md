# Title (Please modify the title)
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

## 2. Components

### Directory

e.g.
```
├── team1
│   └── ....
├── team2
│   └── ....
├── team3
│   └── ....
├── team4
│   └── ....
├── team5
│   └── ....
```

## 3. Data descrption

### Dataset overview
- ![](asset\images\1.png)
- ![](asset\images\2.png)
- 클래스별 이미지 수 불균형 존재
- train/test 데이터 간 분포 차이 극심 (회전각 평균: train 1.9°, test 12.6°)
- 테스트셋은 실제 환경 기반의 비스듬한 촬영, 그림자, 조명 간섭 등 존재

### EDA
- ![](asset\images\3.png)
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
- ![](asset\images\4.png)
- ![](asset\images\5.png)
- ![](asset\images\6.png)

### Presentation
- 발표자료: [PDF 첨부됨]

## etc


### Reference
- https://github.com/facebookresearch/faiss
- https://albumentations.ai/
- https://augraphy.com/
- https://wandb.ai/
- https://pytorch.org/
