import os
from torch.utils.data import Dataset
from PIL import Image

class TestDataset(Dataset):
    """테스트 데이터셋"""
    
    def __init__(self, img_dir: str, transform=None):
        """
        Args:
            img_dir: 이미지 디렉토리 경로
            transform: 이미지 변환 객체
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # 테스트 이미지 파일 목록
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 이미지 파일 경로
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # Transform 적용
        if self.transform:
            image = self.transform(image)
        
        return image, self.img_files[idx] 