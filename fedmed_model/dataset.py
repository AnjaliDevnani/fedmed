# dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from config import cfg

class ChestXRayDataset(Dataset):
    """
    Loads chest X-ray images for binary classification.
    Supports NIH ChestX-ray14 and CheXpert formats.
    """
    def __init__(
        self, 
        csv_path: str,           # CSV with 'image_path' and 'label' cols
        image_dir: str,          # Root directory of images
        transform=None,          # Albumentations transform
        mode: str = "train"      # 'train', 'val', or 'test'
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        # Map labels to binary: 0=Normal, 1=Pneumonia
        # NIH dataset uses "No Finding" for Normal
        label_map = {
            "No Finding": 0, "Normal": 0,
            "Pneumonia": 1, "COVID-19": 1, 
            "Bacterial Pneumonia": 1, "Viral Pneumonia": 1
        }
        self.df['binary_label'] = self.df['label'].map(label_map)
        # Drop rows with unmapped labels
        self.df = self.df.dropna(subset=['binary_label'])
        self.df['binary_label'] = self.df['binary_label'].astype(int)
        
        print(f"[Dataset] {mode}: {len(self.df)} samples")
        print(f"  Normal={sum(self.df.binary_label==0)}, "
              f"Pneumonia={sum(self.df.binary_label==1)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'])
        label = int(row['binary_label'])

        # Load image — handle DICOM and standard formats
        if img_path.endswith('.dcm'):
            image = self._load_dicom(img_path)
        else:
            image = np.array(Image.open(img_path).convert('RGB'))

        # Apply augmentation / normalization
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(label, dtype=torch.long)

    def _load_dicom(self, path: str) -> np.ndarray:
        """Load and convert DICOM to numpy RGB array."""
        import pydicom
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype(np.float32)
        # Normalize to 0-255
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255)
        arr = arr.astype(np.uint8)
        # Convert grayscale to RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    def get_class_weights(self):
        """Compute weights for WeightedRandomSampler to handle imbalance."""
        counts = self.df['binary_label'].value_counts().sort_index()
        total = len(self.df)
        weights_per_class = total / (len(counts) * counts)
        sample_weights = self.df['binary_label'].map(
            weights_per_class.to_dict()
        ).values
        return torch.FloatTensor(sample_weights)


def get_transforms(mode: str = "train"):
    """
    Returns Albumentations transforms for train/val/test.
    More aggressive augmentation for training.
    """
    if mode == "train":
        return A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            # Geometric augmentation
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, 
                rotate_limit=15, p=0.5
            ),
            # Intensity augmentation
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.4
            ),
            # Medical-specific: CLAHE for better contrast
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # Normalize with ImageNet stats
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(),
        ])
    else:  # val / test — NO random augmentation
        return A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.CLAHE(clip_limit=2.0, p=1.0),  # Always apply CLAHE
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(),
        ])


def get_dataloaders(hospital_id: str = None):
    """
    Create train/val/test DataLoaders.
    If hospital_id given, loads that hospital's split.
    Otherwise loads the full dataset.
    """
    if hospital_id:
        # Federated: each hospital has its own split
        train_csv = f"{cfg.split_dir}/hospital_{hospital_id}_train.csv"
        val_csv   = f"{cfg.split_dir}/hospital_{hospital_id}_val.csv"
    else:
        # Central training
        train_csv = f"{cfg.data_dir}/train.csv"
        val_csv   = f"{cfg.data_dir}/val.csv"
    
    train_dataset = ChestXRayDataset(
        train_csv, cfg.data_dir, 
        transform=get_transforms("train"), mode="train"
    )
    val_dataset = ChestXRayDataset(
        val_csv, cfg.data_dir,
        transform=get_transforms("val"), mode="val"
    )
    
    # Weighted sampler to handle class imbalance
    sampler = WeightedRandomSampler(
        weights=train_dataset.get_class_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        sampler=sampler,            # Use weighted sampler, not shuffle
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    
    return train_loader, val_loader
