from pathlib import Path
import json
from collections import defaultdict

import torch
import transformers
import pandas as pd
import sklearn.utils
import torch.utils.data.dataloader
from tqdm import tqdm
from PIL import Image

import meme_entity_detection.utils.task_properties


class MemeRoleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_path: Path,
        tokenizer: str,
        use_faces: bool,
        ocr_type: str,
        description_type: str,
        balance_dataset: bool = False,
    ):
        assert ocr_type == "OCR" or ocr_type == "GPT-4o"
        assert description_type == "GPT-4o" or description_type == "Kosmos-2" or description_type is None

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

        # Deberta: Doesn't work with a slow tokenizer for some reason? Maybe a bug
        self.processor = transformers.AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    def _load_data_into_df(self, file_path: Path) -> pd.DataFrame:
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

    def _correct_gpt4o_classification(self, class_: str, all_roles: list):
        class_ = class_.replace("villian", "villain")
        if not class_ in all_roles:
            class_ = "other"
        return class_

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        upsampled_role_dfs = [self._upsample_and_concat(df, role) for role in ['hero', 'villain', 'victim']]
        return pd.concat([df[df.role == 'other'], *upsampled_role_dfs])

    def _upsample_and_concat(self, df: pd.DataFrame, role: str, n_samples: int = 2000) -> pd.DataFrame:
        df_role = df[df.role == role]
        df_role_upsampled = sklearn.utils.resample(
            df_role,
            replace=True,
            n_samples=n_samples,
            random_state=42,
        )
        return pd.concat([df_role, df_role_upsampled])

    def __getitem__(self, idx):
        image = Image.open(self.image_base_dir / self.data_df['image'].iloc[idx]).convert("RGB")

        row = self.data_df.iloc[idx]
        entity = row["word"]
        faces = " [SEP] " + " - ".join(row["faces"] or []) if self.use_faces else ""

        description = ""
        match self.description_type:
            case "GPT-4o":
                description = " [SEP] " + row["description GPT-4o"]
            case "Kosmos-2":
                description = " [SEP] " + row["description Kosmos-2"]

        match self.ocr_type:
            case "GPT-4o":
                ocr = row["sentence GPT-4o"] or row["sentence"]
            case "OCR":
                ocr = row["sentence"]

        return {
            "image": image,
            "text": entity + " [SEP] " + ocr + " [SEP] " + faces + description,
            "class_id GPT-4o": row["class_id GPT-4o"],
            "label": torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.encoded_labels)

    @staticmethod
    def collate_fn(batch, processor):
        texts = [item['text'] for item in batch]

        # TODO: Check what value makes sense for max_length. One of the paper uses 275, but this significantly slowed down training
        # images = [item['image'] for item in batch]
        # encoding = processor(
        #     text=texts, images=images, return_tensors="pt", padding="max_length", max_length=64, truncation=True
        # )
        # batch_size, height, width = encoding['pixel_mask'].shape
        # encoding['pixel_mask'] = encoding['pixel_mask'].view(batch_size, 1, height, width)

        encoding = processor(texts, truncation=True, padding='max_length', return_tensors="pt", max_length=196)

        encoding['labels'] = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        encoding['class_id GPT-4o'] = torch.tensor([item['class_id GPT-4o'] for item in batch])

        return encoding
