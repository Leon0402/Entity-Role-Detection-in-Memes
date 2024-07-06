from pathlib import Path

import lightning as L
import torch.utils.data
import transformers

from .meme_role_dataset import MemeRoleDataset


class DataModule(L.LightningDataModule):

    def __init__(
        self, data_dir: Path, batch_size: int = 16, seed: int = 42, num_workers: int = 12,
        balance_train_dataset: bool = True, tokenizer: str = "microsoft/deberta-v3-large", use_faces: bool = True,
        ocr_type: str = "OCR", use_gpt_description: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.balance_train_dataset = True  # balance_train_dataset
        self.tokenizer = tokenizer
        self.use_faces = use_faces
        self.ocr_type = ocr_type
        self.use_gpt_description = use_gpt_description

    def setup(self, stage: str):
        self.train_dataset = MemeRoleDataset(
            self.data_dir / "annotations/train.jsonl",
            balance_dataset=self.balance_train_dataset,
            tokenizer=self.tokenizer,
            use_faces=self.use_faces,
            ocr_type=self.ocr_type,
            use_gpt_description=self.use_gpt_description,
        )

        self.validation_dataset = MemeRoleDataset(
            self.data_dir / "annotations/dev.jsonl",
            tokenizer=self.tokenizer,
            use_faces=self.use_faces,
            ocr_type=self.ocr_type,
            use_gpt_description=self.use_gpt_description,
        )

        self.test_dataset = MemeRoleDataset(
            self.data_dir / "annotations/dev_test.jsonl",
            tokenizer=self.tokenizer,
            use_faces=self.use_faces,
            ocr_type=self.ocr_type,
            use_gpt_description=self.use_gpt_description,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.train_dataset.processor),
            pin_memory=True,
            shuffle=True,
            # multiprocessing_context='forkserver'
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.validation_dataset.processor),
            pin_memory=True,
            shuffle=False,
            # multiprocessing_context='forkserver'
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.test_dataset.processor),
            pin_memory=True,
            shuffle=False,
            # multiprocessing_context='forkserver'
        )
