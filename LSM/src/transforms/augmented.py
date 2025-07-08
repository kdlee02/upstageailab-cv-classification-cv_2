import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AugmentedTransform:
    """데이터 증강이 적용된 이미지 변환"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def __call__(self, image):
        """이미지 변환 적용"""
        if isinstance(image, np.ndarray):
            return self.transform(image=image)['image']
        else:
            return self.transform(image=np.array(image))['image']