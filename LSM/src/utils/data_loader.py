import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.basic import BasicDataset
from src.datasets.augmented import AugmentedDataset
from src.datasets.test_dataset import TestDataset
from src.transforms.basic import BasicTransform
from src.transforms.augmented import AugmentedTransform
from PIL import Image
import os
def create_data_loaders(config):
    """설정에 따라 데이터 로더 생성"""
    
    # 변환 생성
    transform_config = config['transform']
    transform_name = transform_config['name']
    
    if transform_name == 'basic':
        transform = BasicTransform(
            image_size=transform_config.get('image_size', 224)
        )
    elif transform_name == 'augmented':
        transform = AugmentedTransform(
            image_size=transform_config.get('image_size', 224)
        )
    else:
        raise ValueError(f"지원하지 않는 변환: {transform_name}")
    
    # 데이터 로드 및 분할
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    
    # 전체 데이터 로드
    full_data = pd.read_csv(dataset_config['train_csv'])
    full_data['path'] = './data/train/'
    if dataset_name == 'augmented':
        os.makedirs(f'./data/augmented/', exist_ok=True)
        traget_count = full_data['target'].value_counts().reset_index(name='count')
        traget_count['augmented_count'] = traget_count['count'].apply(lambda x : 150 - x)
        sample_list = [full_data]
        for target, augmented_count in traget_count[['target','augmented_count']].values:
            target_full_data = full_data[full_data['target'] == target]
            sample = pd.concat([target_full_data.sample(n=1) for _ in range(augmented_count)])
            for path,file in sample[['path','ID']].values:
                image = Image.open(path+file).convert('RGB')
                image = transform(image)
                img_pil = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
                img_pil.save('./data/augmented/' + file)
            sample['path'] = './data/augmented/'
            sample_list.append(sample)
        full_data = pd.concat(sample_list)
        full_data = full_data.sample(frac=1).reset_index(drop=True)
    # train/val 분할
    # train_data, val_data = train_test_split(
    #     full_data, 
    #     test_size=0.2, 
    #     random_state=config.get('seed', 42),
    #     stratify=full_data.iloc[:, 1] if len(full_data.columns) > 1 else None
    # )
    train_data, val_data = train_test_split(
        full_data,
        test_size=0.5,  # 5:5 비율로 설정
        random_state=config.get('seed', 42),
        # 열의 위치 대신 '이름'을 사용하여 stratify 지정 (더 안정적인 방법)
        stratify=full_data['target'] if 'target' in full_data.columns else None
    )
    
    
    # 임시 CSV 파일 생성
    train_csv = 'temp_train.csv'
    val_csv = 'temp_val.csv'
    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)
    
    # 데이터셋 생성
    if dataset_name == 'basic':
        train_dataset = BasicDataset(
            csv_file=train_csv,
            img_dir=dataset_config['train_img_dir'],
            transform=transform
        )
        val_dataset = BasicDataset(
            csv_file=val_csv,
            img_dir=dataset_config['train_img_dir'],
            transform=transform
        )
    elif dataset_name == 'augmented':
        train_dataset = AugmentedDataset(
            csv_file=train_csv,
            img_dir=dataset_config['train_img_dir'],
            transform=transform
        )
        val_dataset = AugmentedDataset(
            csv_file=val_csv,
            img_dir=dataset_config['train_img_dir'],
            transform=transform
        )
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader 