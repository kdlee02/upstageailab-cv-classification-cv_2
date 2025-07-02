import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AugmentedTransform:
    """데이터 증강이 적용된 이미지 변환"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = A.Compose([
            # A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),  # ±30도 자유 회전
            A.RandomBrightnessContrast(p=0.5),  # 밝기/대비
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # 가우시안 노이즈
            A.MotionBlur(blur_limit=5, p=0.3),  # 모션 블러
            A.GridDistortion(p=0.3),  # 종이 찌그러짐처럼
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.3),  # 광학 왜곡
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # ISO 노이즈
            A.ImageCompression(quality_lower=30, quality_upper=70, p=0.3),  # JPEG 압축 아티팩트
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def __call__(self, image):
        """이미지 변환 적용"""
        if isinstance(image, np.ndarray):
            return self.transform(image=image)['image']
        else:
            return self.transform(image=np.array(image))['image']