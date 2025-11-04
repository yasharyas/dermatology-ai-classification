"""
HAM10000 Dataset Class with Advanced Augmentations
Supports multimodal (image + metadata) learning
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset with support for:
    - Image loading and augmentation
    - Metadata features (age, sex, localization)
    - Multimodal fusion
    """
    
    def __init__(
        self,
        csv_file,
        transform=None,
        use_metadata=False,
        img_size=384
    ):
        """
        Args:
            csv_file: Path to train.csv or val.csv
            transform: Albumentations transform pipeline
            use_metadata: Whether to include metadata features
            img_size: Target image size
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_metadata = use_metadata
        self.img_size = img_size
        
        # Create default transform if none provided
        if self.transform is None:
            self.transform = self.get_default_transforms(is_train=False)
        
        # Metadata preprocessing
        if self.use_metadata:
            self._preprocess_metadata()
    
    def _preprocess_metadata(self):
        """Preprocess metadata features"""
        # Age: normalize to [0, 1]
        self.df['age_norm'] = self.df['age'].fillna(self.df['age'].median())
        self.df['age_norm'] = (self.df['age_norm'] - self.df['age_norm'].min()) / \
                              (self.df['age_norm'].max() - self.df['age_norm'].min())
        
        # Sex: one-hot encoding (male, female, unknown)
        sex_dummies = pd.get_dummies(self.df['sex'], prefix='sex')
        for col in ['sex_male', 'sex_female', 'sex_unknown']:
            if col not in sex_dummies.columns:
                sex_dummies[col] = 0
        self.df = pd.concat([self.df, sex_dummies[['sex_male', 'sex_female', 'sex_unknown']]], axis=1)
        
        # Localization: one-hot encoding
        loc_dummies = pd.get_dummies(self.df['localization'], prefix='loc')
        self.loc_cols = loc_dummies.columns.tolist()
        self.df = pd.concat([self.df, loc_dummies], axis=1)
        
        # Total metadata features
        self.metadata_dim = 1 + 3 + len(self.loc_cols)  # age + sex + localization
        print(f"Metadata feature dimension: {self.metadata_dim}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        # Prepare output
        output = {
            'image': image,
            'label': label,
            'image_id': row['image_id']
        }
        
        # Add metadata if requested
        if self.use_metadata:
            metadata_features = [row['age_norm']] + \
                               [row['sex_male'], row['sex_female'], row['sex_unknown']] + \
                               [row[col] for col in self.loc_cols]
            metadata = torch.tensor(metadata_features, dtype=torch.float32)
            output['metadata'] = metadata
        
        return output
    
    @staticmethod
    def get_train_transforms(img_size=384, heavy_aug=False):
        """
        Get training augmentation pipeline
        
        Args:
            img_size: Target image size
            heavy_aug: Use heavy augmentations (slower but more robust)
        """
        if heavy_aug:
            # Heavy augmentations for final training
            return A.Compose([
                A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    scale=(0.85, 1.15),
                    rotate=(-45, 45),
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=1, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.Blur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(img_size // 16, img_size // 8),
                    hole_width_range=(img_size // 16, img_size // 8),
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Lightweight augmentations optimized for RTX 3060 (6GB VRAM)
            return A.Compose([
                A.Resize(img_size, img_size),
                # Fast geometric transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.4
                ),
                # Fast color transforms
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
                # Efficient regularization
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(img_size // 16, img_size // 12),
                    hole_width_range=(img_size // 16, img_size // 12),
                    p=0.25
                ),
                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    @staticmethod
    def get_val_transforms(img_size=384):
        """Get validation transforms (minimal augmentation)"""
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_tta_transforms(img_size=384, n_tta=5):
        """Get Test-Time Augmentation transforms"""
        tta_transforms = []
        
        # Original + 4 augmented versions
        transforms_list = [
            A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            A.Compose([A.Resize(img_size, img_size), A.RandomRotate90(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            A.Compose([A.Resize(img_size, img_size), A.RandomBrightnessContrast(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        ]
        
        return transforms_list[:n_tta]
    
    @staticmethod
    def get_default_transforms(is_train=False, img_size=384):
        """Get default transforms"""
        if is_train:
            return HAM10000Dataset.get_train_transforms(img_size)
        else:
            return HAM10000Dataset.get_val_transforms(img_size)


def get_dataloader(
    csv_file,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    transform=None,
    use_metadata=False,
    img_size=384
):
    """
    Create DataLoader for HAM10000 dataset
    
    Args:
        csv_file: Path to CSV file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        transform: Augmentation pipeline
        use_metadata: Include metadata features
        img_size: Image size
    
    Returns:
        DataLoader object
    """
    dataset = HAM10000Dataset(
        csv_file=csv_file,
        transform=transform,
        use_metadata=use_metadata,
        img_size=img_size
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset
    import json
    
    # Load stats
    with open('d:/CODING/dataverse_files/processed/dataset_stats.json') as f:
        stats = json.load(f)
    
    print("Testing HAM10000Dataset...")
    print(f"Number of classes: {stats['num_classes']}")
    
    # Create dataset
    train_dataset = HAM10000Dataset(
        csv_file='d:/CODING/dataverse_files/processed/train.csv',
        transform=HAM10000Dataset.get_train_transforms(img_size=384),
        use_metadata=True,
        img_size=384
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Test sample
    sample = train_dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")
    if 'metadata' in sample:
        print(f"Metadata shape: {sample['metadata'].shape}")
    
    print("\n✅ Dataset test passed!")
