from pathlib import Path
import json

import torch
import transformers
import torchvision.transforms
import pandas as pd
import sklearn.utils
import torch.utils.data.dataloader
from tqdm import tqdm
from PIL import Image

class MemeRoleDataset(torch.utils.data.Dataset):

    def __init__(self, file_path: Path, balance_dataset: bool = False, use_faces=False):
        self.data_df = self._load_data_into_df(file_path)
        self.use_faces = use_faces

        self.image_base_dir = file_path.parent.parent / "images"

        if balance_dataset:
            self.data_df = self._balance_dataset(self.data_df)

        self.sentences = self.data_df['sentence']
        if self.use_faces:
            self.sentences = self.sentences + " [SEP] - "+ self.data_df["faces"].apply(lambda x: x[0] if x else "")
        self.sentences  = self.sentences.tolist()

        label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}
        self.encoded_labels = [label2id[role] for role in self.data_df['role'].to_list()]

        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((384, 384)), 
            torchvision.transforms.ToTensor()
        ])
    
    def _load_data_into_df(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, 'r') as json_file:
            json_data = [json.loads(line) for line in json_file]

        return pd.DataFrame([{
            "sentence": vals['OCR'].lower().replace('\n', ' '),
            "original": vals['OCR'],
            "faces": vals.get("faces", ""),
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
        image = Image.open(self.image_base_dir / self.data_df['image'].iloc[idx]).convert("RGB")

        return {"image": image, "text": self.sentences[idx], "label": torch.tensor(self.encoded_labels[idx], dtype=torch.long)}
        
    def __len__(self):
        return len(self.encoded_labels)

    @staticmethod
    def collate_fn(batch):
        return {"text":  [item['text'] for item in batch], "image":  [item['image'] for item in batch], "label": torch.utils.data.dataloader.default_collate([item['label'] for item in batch])}
