import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import torchvision.transforms as T
from torchvision.transforms import functional as TF

from timm.data.mixup import Mixup
from augraphy import AugraphyPipeline, VoronoiTessellation
import uuid
import cv2
import os

class SafeVoronoiTessellation(VoronoiTessellation):
    # __init__은 부모 클래스의 것을 그대로 사용하므로 정의할 필요가 없습니다.

    def __call__(self, image, force=False):
        if force or self.should_run():
            # 고유한 임시 파일 경로 생성
            temp_filename = f"temp_voronoi_{uuid.uuid4()}.png"
            
            try:
                # Augraphy의 원래 로직을 거의 그대로 따르되, 파일 이름만 변경합니다.
                # (이 코드는 Augraphy 라이브러리의 원본 코드를 기반으로 합니다)
                mask = self.create_voronoi(image)
                
                # 원본 라이브러리가 임시 파일을 저장하고 다시 읽는 로직을 사용
                cv2.imwrite(temp_filename, mask)
                voronoi_img = cv2.imread(temp_filename)

                # 그레이스케일 이미지에 컬러를 적용해야 할 경우
                if self.multichannel:
                    voronoi_img = cv2.cvtColor(voronoi_img, cv2.COLOR_GRAY2BGR)

                return voronoi_img

            finally:
                # 작업이 성공하든 실패하든 임시 파일은 반드시 삭제
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        else:
            return image
class ResizeWithPadding:
    """
    이미지 비율을 유지하며 리사이즈하고, 남는 공간을 패딩으로 채웁니다.
    """
    def __init__(self, size, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        w, h = image.size
        scale = min(self.size[0] / w, self.size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = TF.resize(image, (new_h, new_w))
        pad_left = (self.size[0] - new_w) // 2
        pad_top = (self.size[1] - new_h) // 2
        pad_right = self.size[0] - new_w - pad_left
        pad_bottom = self.size[1] - new_h - pad_top
        return TF.pad(resized, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

class AugmentedTransform:
    """데이터 증강이 적용된 이미지 변환"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

        self.transform = T.Compose([
            ResizeWithPadding((self.image_size, self.image_size), fill=(255, 255, 255)),
        ])
        
        # VoronoiTessellation을 직접 사용
        self.voronoi_transform = SafeVoronoiTessellation(num_cells_range=(150, 250), p=0.5)
        
        # 더 안전한 albumentations 변환들
        self.albumentations_transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Rotate(limit=360, p=0.5, border_mode=0, value=(255, 255, 255)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    def __call__(self, image):
        """이미지 변환 적용"""
        # 1. 초기 리사이즈 및 패딩 (입력: PIL, 출력: PIL)
        # ResizeWithPadding이 먼저 적용되어 이미지 크기를 통일합니다.
        resized_image = self.transform(image)
        
        # 2. PIL 이미지를 NumPy 배열로 변환 (Augraphy/Albumentations 입력용)
        image_np = np.array(resized_image)
        
        # 3. VoronoiTessellation 직접 적용 (입력: NumPy, 출력: NumPy)
        # Voronoi Tessellation 등 문서 품질 저하 효과를 적용합니다.
        # try:
        #     voronoi_applied = self.voronoi_transform(image=image_np)
        # except Exception as e:
        #     # 멀티프로세스 환경에서 파일 충돌 등으로 인한 에러 무시
        #     voronoi_applied = image_np

        # 4. Albumentations 파이프라인 적용 (입력: NumPy, 출력: 텐서)
        # 기하학/색상 증강, 정규화, 텐서 변환을 최종적으로 적용합니다.
        # Albumentations는 딕셔너리를 반환하므로 'image' 키로 값을 추출해야 합니다.
        final_tensor = self.albumentations_transform(image=image_np)['image']
        
        return final_tensor