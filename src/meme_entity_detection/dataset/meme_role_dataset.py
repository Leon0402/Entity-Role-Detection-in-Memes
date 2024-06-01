from pathlib import Path
import json

import torch
import transformers
import pandas as pd
import sklearn.utils
from tqdm import tqdm


class MemeRoleDataset(torch.utils.data.Dataset):

    def __init__(self, file_path: Path, balance_dataset: bool = False):
        self.data_df = self._load_data_into_df(file_path)
        # TODO: balancing logic seems slightly weird, maybe double check, but it was used like this in the original code
        if balance_dataset:
            self.data_df = self._balance_dataset(self.data_df)

        tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)
        self.text_encodings = tokenizer(
            self.data_df['sentence'].to_list(), self.data_df['word'].to_list(), truncation=True, padding='max_length',
            max_length=64
        )

        label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}
        self.encoded_labels = [label2id[role] for role in self.data_df['role'].to_list()]

    def _load_data_into_df(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, 'r') as json_file:
            json_data = [json.loads(line) for line in json_file]

        return pd.DataFrame([{
            "sentence": vals['OCR'].lower().replace('\n', ' '),
            "original": vals['OCR'],
            "word": word_val,
            "image": vals['image'],
            "role": role
        } for vals in tqdm(json_data) for role in ['hero', 'villain', 'victim', 'other'] for word_val in vals[role]])

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
        return {
            'input_ids': torch.tensor(self.text_encodings.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.text_encodings.attention_mask[idx], dtype=torch.long),
            'token_type_ids': torch.tensor(self.text_encodings.token_type_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.encoded_labels)
