import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import ast
class Augmented_V2_Dataset(Dataset):
    """데이터 증강이 적용된 문서 분류 데이터셋"""
    
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        """
        Args:
            csv_file: CSV 파일 경로
            img_dir: 이미지 디렉토리 경로
            transform: 이미지 변환 객체 (AugmentedTransform 등)
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 이미지 파일명 가져오기
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if len(self.data.columns) > 1:
            label = torch.tensor(self.data.iloc[idx, 1])
            
            return image, label
        else:
            return image