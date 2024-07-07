from pathlib import Path
import json
from collections import defaultdict

import torch
import pandas as pd
import sklearn.utils
import torch.utils.data.dataloader
from tqdm import tqdm
from PIL import Image

import meme_entity_detection.utils.task_properties
import meme_entity_detection.model.interface

from .config import OcrType, DescriptionType


class MemeRoleDataset(torch.utils.data.Dataset):
    """
    Dataset class for Meme Role Detection.
    """

    def __init__(
        self,
        file_path: Path,
        use_faces: bool,
        ocr_type: OcrType,
        description_type: DescriptionType,
        balance_dataset: bool = False,
    ):
        """
        Initialize the MemeRoleDataset.

        Parameters:
            file_path: Path to the dataset json file.
            use_faces: Whether to use recognized celebrity faces in dataset.
            ocr_type: Type of OCR to use.
            description_type: Type of description to use.
            balance_dataset: Whether to balance the training dataset (e.g. increase number of heroes, villain & victim samples by resampling, which are underpresented)
        """

        self.data_df = self._load_data_into_df(file_path)
        self.use_faces = use_faces
        self.ocr_type = ocr_type
        self.description_type = description_type

        self.image_base_dir = file_path.parent.parent / "images"

        if balance_dataset:
            self.data_df = self._balance_dataset(self.data_df)

        self.encoded_labels = [
            meme_entity_detection.utils.task_properties.label2id[role] for role in self.data_df['role'].to_list()
        ]

    def _load_data_into_df(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from the given file path into a DataFrame.

        Parameters:
            file_path: Path to the json dataset file.

        Returns:
            DataFrame containing the dataset information.
        """
        with open(file_path, 'r') as json_file:
            json_data = [json.loads(line) for line in json_file]

        all_roles = ['hero', 'villain', 'victim', 'other']
        df = pd.DataFrame([{
            "sentence": vals['OCR'].lower().replace('\n', ' '),
            "sentence GPT-4o": (vals["OCR GPT-4o"] or "").lower().replace('\n', ' '),
            "description GPT-4o": (vals["IMAGE DESCRIPTION GPT-4o"] or "").lower().replace('\n', ' '),
            "description Kosmos-2": (vals["Kosmos Image Descriptions"] or "").lower().replace('\n', ' '),
            "classification GPT-4o": defaultdict(lambda: "other", vals["CLASSIFICATION GPT-4o"])
            if vals["CLASSIFICATION GPT-4o"] else defaultdict(lambda: "other"),
            "original": vals['OCR'],
            "faces": vals.get("faces", ""),
            "word": word_val,
            "image": vals['image'],
            "role": role
        } for vals in tqdm(json_data) for role in all_roles for word_val in vals[role]])

        df["classification GPT-4o"] = [
            data["classification GPT-4o"][data["word"]]
            for _, data in tqdm(df[["classification GPT-4o", "word"]].iterrows())
        ]
        df["classification GPT-4o"] = df["classification GPT-4o"].str.replace("villian", "villain")
        df["classification GPT-4o"] = df["classification GPT-4o"].apply(
            lambda x: self._correct_gpt4o_classification(x, all_roles)
        )
        df["class_id GPT-4o"] = df["classification GPT-4o"].apply(
            lambda x: meme_entity_detection.utils.task_properties.label2id[x]
        )

        return df

    def _correct_gpt4o_classification(self, role: str, all_roles: list):
        """
        Correct the classification if it is not in the list of all roles.

        Parameters:
            role: The role classified by ChatGPT-4o.
            all_roles: List of all possible roles.

        Returns:
            Corrected classification.
        """

        role = role.replace("villian", "villain")
        if role not in all_roles:
            role = "other"
        return role

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset by resampling underrepresented classes.

        Parameters:
            df: DataFrame containing the dataset information.

        Returns:
            Balanced DataFrame.
        """
        upsampled_role_dfs = [self._resample_and_concat(df, role) for role in ['hero', 'villain', 'victim']]
        return pd.concat([df[df.role == 'other'], *upsampled_role_dfs])

    def _resample_and_concat(self, df: pd.DataFrame, role: str, n_samples: int = 2000) -> pd.DataFrame:
        """
        Resample the specified role in the dataset.

        Parameters:
            df: DataFrame containing the dataset information.
            role: The role to resample.
            n_samples: Number of samples to upsample to.

        Returns:
            DataFrame with the resampled role.
        """
        df_role = df[df.role == role]
        df_role_upsampled = sklearn.utils.resample(
            df_role,
            replace=True,
            n_samples=n_samples,
            random_state=42,
        )
        return pd.concat([df_role, df_role_upsampled])

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item from the dataset by index.

        NOTE: This function is overriden and automatically called by the pytorch DataLoader.

        Parameters:
            idx: Index of the item to retrieve.

        Returns:
            Dictionary containing the image, text, predicted label by GPT-4o, and label.
        """
        image = Image.open(self.image_base_dir / self.data_df['image'].iloc[idx]).convert("RGB")

        row = self.data_df.iloc[idx]
        entity = row["word"]

        match self.ocr_type:
            case OcrType.STANDARD:
                ocr = " [SEP] " + row["sentence"]
            case OcrType.GPT:
                ocr = " [SEP] " + row["sentence GPT-4o"] or row["sentence"]
            case OcrType.NONE:
                ocr = ""

        match self.description_type:
            case DescriptionType.KOSMOS:
                description = " [SEP] " + row["description Kosmos-2"]
            case DescriptionType.GPT:
                description = " [SEP] " + row["description GPT-4o"]
            case DescriptionType.NONE:
                description = ""

        faces = ""
        if self.use_faces:
            faces = " [SEP] " + " - ".join(row["faces"] or [])

        return {
            "image": image,
            "text": entity + ocr + faces + description,
            "class_id GPT-4o": row["class_id GPT-4o"],
            "label": torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.encoded_labels)

    @staticmethod
    def collate_fn(batch: dict, processor: meme_entity_detection.model.interface.Tokenizer):
        """
        Collate function to process a batch of data.

        Parameters:
            batch: Batch of data to collate.
            processor: Tokenizer processor to use.

        Returns:
            Dictionary containing the processed batch data.
        """
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]

        encoding = processor.tokenize(texts, images)
        encoding['labels'] = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        encoding['class_id GPT-4o'] = torch.tensor([item['class_id GPT-4o'] for item in batch])

        return encoding
