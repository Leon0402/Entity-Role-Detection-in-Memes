from pathlib import Path

import lightning as L
import torch.utils.data
import transformers

from .meme_role_dataset import MemeRoleDataset


class DataModule(L.LightningDataModule):

    def __init__(self, data_dir: Path, batch_size: int = 16, seed: int = 42, num_workers: int = 8, balance_train_dataset: bool = True, use_faces: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.balance_train_dataset = balance_train_dataset
        self.use_faces = use_faces

        self.processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

    def setup(self, stage: str):
        self.train_dataset = MemeRoleDataset(self.data_dir / "annotations/train.jsonl", balance_dataset=self.balance_train_dataset, use_faces=self.use_faces)
        self.validation_dataset = MemeRoleDataset(self.data_dir / "annotations/dev.jsonl", use_faces=self.use_faces)
        self.test_dataset = MemeRoleDataset(self.data_dir / "annotations/dev_test.jsonl", use_faces=self.use_faces)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.processor),
            pin_memory=True,
            shuffle=True,
            # multiprocessing_context='forkserver'
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.processor),
            pin_memory=True,
            shuffle=False,
            # multiprocessing_context='forkserver'
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.processor),
            pin_memory=True,
            shuffle=False,
            # multiprocessing_context='forkserver'
        )
