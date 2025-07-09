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
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.meshgenerator import Noise
from augraphy.utilities.slidingwindow import PatternMaker
from numba import config, jit
import random
class SafeVoronoiTessellation(Augmentation):
    """
    파일 I/O 없이 Voronoi Tessellation을 수행하는 Safe 버전입니다.
    Augraphy의 VoronoiTessellation 내부 로직에서 generate_voronoi를 그대로 사용하며,
    파일 저장/읽기 단계를 제거했습니다.
    """
    def __init__(
        self,
        mult_range=(50, 80),
        seed=19829813472,
        num_cells_range=(500, 1000),
        noise_type="random",
        background_value=(200, 255),
        numba_jit=1,
        p=1.0,
    ):
        super().__init__(p=p)
        self.mult_range = mult_range
        self.seed = seed
        self.num_cells_range = num_cells_range
        self.noise_type = noise_type
        self.background_value = background_value
        config.DISABLE_JIT = bool(1 - numba_jit)

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def generate_voronoi(width, height, num_cells, nsize, pixel_data, perlin_noise_2d):
        """
        Voronoi mesh를 생성하는 numba 가속 staticmethod
        """
        img_array = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                dmin = np.hypot(height, width)
                idx = 0
                for i in range(num_cells):
                    dx = pixel_data[0][i] - x + perlin_noise_2d[0][y][x]
                    dy = pixel_data[1][i] - y + perlin_noise_2d[1][y][x]
                    d = np.hypot(dx, dy)
                    if d < dmin:
                        dmin = d
                        idx = i
                    nsize[idx] += 1
                img_array[y, x] = pixel_data[2][idx]
        return img_array

    def apply_augmentation(self):
        # Perlin noise 준비
        obj_noise = Noise()
        perlin_x = np.zeros((self.height, self.width), dtype=np.float32)
        perlin_y = np.zeros((self.height, self.width), dtype=np.float32)
        if self.perlin:
            for y in range(self.height):
                for x in range(self.width):
                    perlin_x[y, x] = obj_noise.noise2D(x / 100, y / 100) * self.mult
                    perlin_y[y, x] = obj_noise.noise2D((x + self.seed) / 100, (y + self.seed) / 100) * self.mult

        # 랜덤 포인트 및 색상
        num_cells = random.randint(*self.num_cells_range)
        xs = [random.randrange(self.width) for _ in range(num_cells)]
        ys = [random.randrange(self.height) for _ in range(num_cells)]
        colors = [random.randrange(self.background_value[0], self.background_value[1]) for _ in range(num_cells)]
        sizes = np.zeros(num_cells, dtype=np.int32)

        # Voronoi mesh 생성
        mesh = SafeVoronoiTessellation.generate_voronoi(
            self.width, self.height, num_cells, sizes,
            (xs, ys, colors), (perlin_x, perlin_y)
        )
        return mesh

    def __call__(
        self,
        image,
        layer=None,
        mask=None,
        keypoints=None,
        bounding_boxes=None,
        force=False,
        **kwargs
    ):
        # 실행 여부 결정
        if not (force or self.should_run()):
            return image

        # 입력 복사 및 알파 분리
        result = image.copy()
        alpha = None
        if result.ndim == 3 and result.shape[2] == 4:
            alpha = result[:, :, 3]
            result = result[:, :, :3]

        # noise_type 처리
        self.perlin = random.choice([True, False]) if self.noise_type == 'random' else (self.noise_type == 'perlin')

        # 메쉬 크기 결정
        size_choices = [100, 120, 140, 160, 180, 200] if self.perlin else [200, 210, 220, 240, 260, 280]
        self.width = self.height = random.choice(size_choices)
        divisors = [50, 70, 80, 90] if self.perlin else [100, 120, 140, 150]
        self.ws = next((d for d in divisors if self.width % d == 0), divisors[0])

        # 파라미터 설정
        self.mult = random.randint(*self.mult_range)
        self.num_cells = random.randint(*self.num_cells_range)

        # Voronoi mesh 생성 및 리사이즈
        vor_mesh = self.apply_augmentation()
        vor_mesh = cv2.resize(vor_mesh, (self.ws, self.ws), interpolation=cv2.INTER_LINEAR)

        # 채널 일치
        if result.ndim == 2 and vor_mesh.ndim == 3:
            vor_mesh = cv2.cvtColor(vor_mesh, cv2.COLOR_RGB2GRAY)
        elif result.ndim == 3 and vor_mesh.ndim == 2:
            vor_mesh = cv2.cvtColor(vor_mesh, cv2.COLOR_GRAY2BGR)

        # sliding window 패턴 적용
        sw = PatternMaker()
        padded = sw.make_patterns(result, vor_mesh, self.ws)
        h, w = result.shape[:2]
        result = padded[self.ws:h+self.ws, self.ws:w+self.ws]

        # 알파 복원
        if alpha is not None:
            result = np.dstack((result, alpha))

        # 추가 출력 지원
        if mask is not None or keypoints is not None or bounding_boxes is not None:
            return [result, mask, keypoints, bounding_boxes]
        return result



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
        self.pipeline = AugraphyPipeline([
            SafeVoronoiTessellation(num_cells_range=(500, 800), p=0.5),
        ])
        
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

        aug_bgr = self.pipeline(image=image_np)
        # aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)
        # 4. Albumentations 파이프라인 적용 (입력: NumPy, 출력: 텐서)
        # 기하학/색상 증강, 정규화, 텐서 변환을 최종적으로 적용합니다.
        # Albumentations는 딕셔너리를 반환하므로 'image' 키로 값을 추출해야 합니다.
        final_tensor = self.albumentations_transform(image=aug_bgr)['image']
        
        return final_tensor