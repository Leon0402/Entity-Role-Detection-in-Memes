from pathlib import Path

import lightning as L
import torch.utils.data

import meme_entity_detection.model.interface
import meme_entity_detection.model.deberta

from .meme_role_dataset import MemeRoleDataset
from .config import OcrType, DescriptionType


class DataModule(L.LightningDataModule):
    """
    DataModule for loading and preparing data for meme entity detection.
    """

    def __init__(
        self, data_dir: Path, batch_size: int = 16, seed: int = 42, num_workers: int = 12,
        balance_train_dataset: bool = True,
        tokenizer: meme_entity_detection.model.interface.Tokenizer = meme_entity_detection.model.DebertaTokenizer(),
        use_faces: bool = True, ocr_type: OcrType = OcrType.STANDARD,
        description_type: DescriptionType = DescriptionType.NONE
    ):
        """
        Initialize the DataModule with given parameters.

        Parameters:
            data_dir: Directory where the data is stored.
            batch_size: Size of the batches for data loading.
            seed: Seed for random number generation.
            num_workers: Number of worker processes for data loading.
            balance_train_dataset: Whether to balance the training dataset (e.g. increase number of heroes, villain & victim samples by resampling, which are underpresented)
            tokenizer: Tokenizer to be used.
            use_faces: Whether to use recognized celebrity faces in dataset.
            ocr_type: Type of OCR to use.
            description_type: Type of description to use.
        """

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.balance_train_dataset = balance_train_dataset
        self.tokenizer = tokenizer
        self.use_faces = use_faces
        self.ocr_type = ocr_type
        self.description_type = description_type

    def setup(self, stage: str) -> None:
        """
        Set up the datasets for training, validation, and testing.

        Note: This function is overriden and automatically called by pytorch lightning.

        Parameters:
            stage: Stage of the setup process (e.g., 'fit', 'validate', 'test', 'predict').
        """
        self.train_dataset = MemeRoleDataset(
            self.data_dir / "annotations/train.jsonl",
            balance_dataset=self.balance_train_dataset,
            use_faces=self.use_faces,
            ocr_type=self.ocr_type,
            description_type=self.description_type,
        )

        self.validation_dataset = MemeRoleDataset(
            self.data_dir / "annotations/dev.jsonl",
            use_faces=self.use_faces,
            ocr_type=self.ocr_type,
            description_type=self.description_type,
        )

        self.test_dataset = MemeRoleDataset(
            self.data_dir / "annotations/dev_test.jsonl",
            use_faces=self.use_faces,
            ocr_type=self.ocr_type,
            description_type=self.description_type,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the DataLoader for the training dataset.

        NOTE: This function is overriden and automatically called by pytorch lightning.

        Returns:
            DataLoader for the training dataset.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.tokenizer),
            pin_memory=True,
            shuffle=True,
            # multiprocessing_context='forkserver'
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the DataLoader for the validation dataset.

        NOTE: This function is overriden and automatically called by pytorch lightning.

        Returns:
            DataLoader for the validation dataset.
        """
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.tokenizer),
            pin_memory=True,
            shuffle=False,
            # multiprocessing_context='forkserver'
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the DataLoader for the test dataset.

        NOTE: This function is overriden and automatically called by pytorch lightning.

        Returns:
            DataLoader for the test dataset.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: MemeRoleDataset.collate_fn(batch, self.tokenizer),
            pin_memory=True,
            shuffle=False,
            # multiprocessing_context='forkserver'
        )
