import os
from pathlib import Path
from typing import Tuple

import cv2
import albumentations as A
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotate,
    ColorJitter, RandomBrightnessContrast, CLAHE,
    GaussianBlur, CoarseDropout, Resize, Normalize
)
from PIL import Image
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split

from src.datasets.datasets import ImageDataset
from augraphy import *
import random

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
        VoronoiTessellation(
            mult_range=(80, 120),               
            num_cells_range=(800, 1500),        
            noise_type="random",
            background_value=(180, 230),        
        ),
    ], p=1.0),
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

import albumentations as A

class AugraphyAlbumentationsWrapper(A.ImageOnlyTransform):
    def __init__(self, augraphy_pipeline, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.augraphy_pipeline = augraphy_pipeline

    def apply(self, img, **params):
        return self.augraphy_pipeline(img)

class DatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.data.num_workers
        self.batch_size = cfg.data.batch_size
        self.img_size = cfg.data.img_size
        self.data_path = cfg.data.data_path

        augraphy_pipeline = get_augraphy_pipeline()
        train_tf =  Compose([
                            Resize(self.img_size, self.img_size),
                            HorizontalFlip(p=0.5),
                            VerticalFlip(p=0.5),
                            Rotate(limit=180, p=1),
                            AugraphyAlbumentationsWrapper(augraphy_pipeline, p=1.0),
                            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])

        self.train_tf = train_tf
        self.val_tf = train_tf

        self.test_tf = Compose(
            [
                Resize(self.img_size, self.img_size),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        self.train_idx = None
        self.val_idx = None
        self.train_df = None
        self.val_df = None

        print("Using Augraphy")
        self.dataset_cls = ImageDataset
        self.full_data_name = "train"
        self.full_df = pd.read_csv(os.path.join(self.data_path, self.full_data_name + ".csv"))

        self.meta_df = pd.read_csv(os.path.join(self.data_path, "meta.csv"))
        self.orig_df = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        self.origin_dataset = self.dataset_cls(self.orig_df, os.path.join(self.data_path, self.full_data_name), None)
    
    def set_split_idx(self, train_idx, val_idx):
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage: str | None = None):
        if stage in ("train", None):
            if self.train_idx is not None and self.val_idx is not None:

                train_df_orig = self.orig_df.iloc[self.train_idx].reset_index(drop=True)
                val_df = self.orig_df.iloc[self.val_idx].reset_index(drop=True)

                train_ids = set(train_df_orig["ID"])
                aug_df = self.full_df[
                    self.full_df["ID"].str.startswith("aug_") &
                    self.full_df["ID"].str[4:].isin(train_ids)
                ].reset_index(drop=True)

                self.train_df = pd.concat([train_df_orig, aug_df]).reset_index(drop=True)
                self.val_df = val_df

                assert self.val_df["ID"].str.startswith("aug_").sum() == 0
            else:
                train_idx, val_idx = train_test_split(
                    self.orig_df.index, test_size=0.2, stratify=self.orig_df["target"], random_state=42
                )
                train_df_orig = self.orig_df.iloc[train_idx].reset_index(drop=True)
                val_df = self.orig_df.iloc[val_idx].reset_index(drop=True)

                train_ids = set(train_df_orig["ID"])
                aug_df = self.full_df[
                    self.full_df["ID"].str.startswith("aug_") &
                    self.full_df["ID"].str[4:].isin(train_ids)
                ].reset_index(drop=True)

                self.train_df = pd.concat([train_df_orig, aug_df]).reset_index(drop=True)
                self.val_df = val_df

                assert self.val_df["ID"].str.startswith("aug_").sum() == 0
                
                self.set_split_idx(train_idx, val_idx)

            self.train_ds = self.dataset_cls(
                self.train_df, os.path.join(self.data_path, self.full_data_name), transform=self.train_tf
            )
            self.val_ds = self.dataset_cls(
                self.val_df, os.path.join(self.data_path, self.full_data_name), transform=self.val_tf
            )

        if stage in ("test", "predict", None):
            df = pd.read_csv(os.path.join(self.data_path, "sample_submission.csv"))
            self.test_ds = self.dataset_cls(
                df, os.path.join(self.data_path, "test"), transform=self.test_tf, is_test=True
            )

    def set_train_dataset(self, new_df):
        self.train_ds = self.dataset_cls(
            new_df, 
            os.path.join(self.data_path, self.full_data_name),
            transform=self.train_tf,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )