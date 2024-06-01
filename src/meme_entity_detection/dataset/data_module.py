from pathlib import Path

import lightning as L
import torch.utils.data

from .meme_role_dataset import MemeRoleDataset


class DataModule(L.LightningDataModule):

    def __init__(self, data_dir: Path, batch_size: int = 16, seed: int = 42, num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset = MemeRoleDataset(self.data_dir / "annotations/train.jsonl", balance_dataset=True)
        self.validation_dataset = MemeRoleDataset(self.data_dir / "annotations/dev.jsonl")
        self.test_dataset = MemeRoleDataset(self.data_dir / "annotations/dev_test.jsonl")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
