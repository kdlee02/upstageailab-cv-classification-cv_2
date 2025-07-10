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
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotate,
    ColorJitter, RandomBrightnessContrast, CLAHE,
    GaussianBlur, CoarseDropout, Resize, Normalize
)
import torch
import kornia.augmentation as K
from augraphy import *
from torchvision.transforms import ToTensor
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
        # numba JIT 설정은 기본값으로 유지

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
paper_phase = [
    OneOf([
        DelaunayTessellation(
            n_points_range=(500, 800),
            n_horizontal_points_range=(500, 800),
            n_vertical_points_range=(500, 800),
            noise_type="random",
            color_list="default",
            color_list_alternate="default",
        ),
        PatternGenerator(
            imgx=random.randint(256, 512),
            imgy=random.randint(256, 512),
            n_rotation_range=(10, 15),
            color="random",
            alpha_range=(0.35, 0.7), 
        ),
        SafeVoronoiTessellation(
            mult_range=(80, 120),               
            num_cells_range=(800, 1500),        
            noise_type="random",
            background_value=(180, 230),        
        ),
    ], p=1),
    AugmentationSequence([
        NoiseTexturize(
            sigma_range=(20, 30),
            turbulence_range=(8, 15),          
        ),
        BrightnessTexturize(
            texturize_range=(0.75, 0.9),       
            deviation=0.08,                    
        ),
    ]),
]

def get_augraphy_pipeline():
    return AugraphyPipeline(paper_phase=paper_phase)

class AugraphyAlbumentationsWrapper(A.ImageOnlyTransform):
    def __init__(self, augraphy_pipeline, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.augraphy_pipeline = augraphy_pipeline

    def apply(self, img, **params):
        return self.augraphy_pipeline(img)





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
        resized = TF.resize(image, [new_h, new_w])
        pad_left = (self.size[0] - new_w) // 2
        pad_top = (self.size[1] - new_h) // 2
        pad_right = self.size[0] - new_w - pad_left
        pad_bottom = self.size[1] - new_h - pad_top
        return TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

class AugmentedTransform:
    """데이터 증강이 적용된 이미지 변환"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

        augraphy_pipeline = get_augraphy_pipeline()
        self.train_tf =  Compose([
                            Resize(self.image_size, self.image_size),
                            HorizontalFlip(p=0.5),
                            VerticalFlip(p=0.5),
                            Rotate(limit=180, p=1),
                            AugraphyAlbumentationsWrapper(augraphy_pipeline, p=1.0),
                            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])


        train_tf =  Compose([
            # 크롭/회전/플립
            RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.8, 1.0), p=1.0),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.3),
            Rotate(limit=15, p=0.5),

            # 색상 변화 (RandAugment 대체 조합)
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # RandomContrast(limit=0.2, p=0.3),
            CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

            # 블러 / 지우기
            GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=0.3),
            CoarseDropout(
                max_holes=1,
                max_height=int(self.image_size * 0.2),
                max_width=int(self.image_size * 0.2),
                fill_value=0,
                p=0.25
            ),

            # 정규화 + Tensor
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_tf = {"Augraphy" : 
                         Compose([
                            # 크롭/회전/플립
                            RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.8, 1.0), p=1.0),
                            HorizontalFlip(p=0.5),
                            VerticalFlip(p=0.3),
                            Rotate(limit=15, p=0.5),
                            
                            # 정규화 + Tensor
                            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()]), 
                        "NoAugraphy": train_tf}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        no_augraphy_tf = torch.nn.Sequential(
            # 크롭/회전/플립
            K.RandomResizedCrop(size=(self.image_size, self.image_size),
                                scale=(0.8, 1.0),
                                ratio=(0.75, 1.3333),
                                p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.3),
            K.RandomRotation(degrees=15.0, p=0.5),

            # 색상 변화
            K.ColorJitter(brightness=(0.8, 1.2),
                          contrast=(0.8, 1.2),
                          saturation=(0.8, 1.2),
                          hue=(-0.2, 0.2),
                          p=0.5),

            # 블러
            K.RandomGaussianBlur(kernel_size=(3, 3),
                                 sigma=(0.1, 2.0),
                                 p=0.3),

            # 랜덤 지우기 (CoarseDropout 대체)
            K.RandomErasing(scale=(0.02, 0.2),
                            ratio=(0.3, 3.3),
                            value=0,
                            p=0.25),

            # 정규화
            K.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ).to(self.device)

        # “Augraphy” 파이프라인: 간단 증강 + 정규화
        augraphy_tf = torch.nn.Sequential(
            K.RandomResizedCrop(size=(self.image_size, self.image_size),
                                scale=(0.8, 1.0),
                                ratio=(0.75, 1.3333),
                                p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.3),
            K.RandomRotation(degrees=15.0, p=0.5),

            K.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ).to(self.device)

        self.train_tf = {
            "Augraphy": augraphy_tf,
            "NoAugraphy": no_augraphy_tf
        }
        
    def __call__(self, image):
        """PIL.Image 또는 HWC NumPy → (C,H,W) Tensor → GPU 증강 → (C,H,W) Tensor 반환"""
        # 1) PIL → Tensor, 또는 NumPy(H×W×C uint8) → Tensor
        if hasattr(image, 'convert'):
            img_t = ToTensor()(image)  # C×H×W, float [0,1]
        else:
            img_np = image  # assume H×W×C uint8 or float[0,255]
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0)

        # 2) 배치 차원 & device 이동
        img_t = img_t.unsqueeze(0).to(self.device, non_blocking=True)

        # 3) 10% 확률로 “Augraphy” 사용
        key = "Augraphy" if (random.random() < 0.1) else "NoAugraphy"
        pipeline = self.train_tf[key]

        # 4) GPU 증강 파이프라인 적용
        out = pipeline(img_t)

        # 5) 배치 제거 후 반환
        return out.squeeze(0)