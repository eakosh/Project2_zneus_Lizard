import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

from dataset import LizardDataset
from transforms import get_train_transforms, get_val_transforms


class LizardPatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        stain_reference_path: str,
        patch_size: int = 256,
        stride: Optional[int] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.data_root = data_root
        self.stain_reference_path = stain_reference_path
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transform = get_train_transforms()
        self.val_transform = get_val_transforms()

    def setup(self):
        self.train_dataset = LizardDataset(
                self.data_root, 1, self.patch_size, self.stride,
                self.stain_reference_path, transform=self.train_transform
            )
        
        self.val_dataset = LizardDataset(
                self.data_root, 2, self.patch_size, self.stride,
                self.stain_reference_path, transform=self.val_transform
            )

        self.test_dataset = LizardDataset(
                self.data_root, 3, self.patch_size, self.stride,
                self.stain_reference_path, transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
