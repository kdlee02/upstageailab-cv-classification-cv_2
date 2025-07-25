import os
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from augraphy import *
import random

class ImageDataset(Dataset):
    def __init__(self, data, path, transform=None, is_test=False):
        if isinstance(data, (str, Path)):
            self.df = pd.read_csv(data).values
        else:
            self.df = np.array(data)  # Convert to numpy array first
            
        # Ensure targets are integers
        if len(self.df) > 0:
            # If it's a 1D array, reshape it to 2D
            if len(self.df.shape) == 1:
                self.df = self.df.reshape(-1, 1)
                
            # Ensure we have at least 2 columns
            if self.df.shape[1] < 2:
                raise ValueError(f"Input data must have at least 2 columns, got {self.df.shape[1]}")
                
            # Convert second column to integers
            try:
                self.df = np.column_stack((self.df[:, 0], self.df[:, 1].astype(int)))
            except Exception as e:
                raise ValueError(f"Failed to convert labels to integers: {e}")
            
        self.path = path
        self.transform = transform
        self.is_test = is_test
        
        # Print dataset statistics
        if len(self.df) > 0:
            try:
                unique_labels = np.unique(self.df[:, 1])
                print(f"Dataset initialized with {len(self.df)} samples and {len(unique_labels)} classes")
                print(f"Label range: {unique_labels.min()} to {unique_labels.max()}")
            except Exception as e:
                print(f"Warning: Could not calculate dataset statistics: {str(e)}")
            
        self.samples = []
        if not is_test:
            for name, target in self.df:
                self.samples.append((name, int(target)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        target = int(target)  # Ensure target is an integer
        
        try:
            img = np.array(Image.open(os.path.join(self.path, name)).convert('RGB'))
            if self.transform:
                img = self.transform(image=img)['image']
            return img, target
        except Exception as e:
            print(f"Error loading image {name}: {str(e)}")
            # Return a zero image and -1 as label in case of error
            if self.transform:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                img = self.transform(image=img)['image']
                return img, -1
            raise