import torch
import transformers
from pathlib import Path
import json
import pandas as pd
import sklearn.utils
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

class MemeRoleDataset(torch.utils.data.Dataset):

    def __init__(self, file_path: Path, balance_dataset: bool = False, text_tokenizer: str = "microsoft/deberta-v3-large", image_tokenizer: str = "google/vit-base-patch16-224", use_faces=False):
        self.data_df = self._load_data_into_df(file_path)
        self.use_faces = use_faces

        if balance_dataset:
            self.data_df = self._balance_dataset(self.data_df)

        self.text_tokenizer = transformers.AutoTokenizer.from_pretrained(text_tokenizer, use_fast=False)
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(image_tokenizer)

        self.sentences = self.data_df['sentence']
        self.image_paths = self.data_df['image']
        
        if self.use_faces:
            self.sentences = self.sentences + " [SEP] - "+ self.data_df["faces"].apply(lambda x: x[0] if x else "")

        self.encodings = self.text_tokenizer(
            self.sentences.to_list(),
            self.data_df['word'].to_list(),
            truncation=True,
            padding='max_length',
            max_length=64
        )

        label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}
        self.encoded_labels = [label2id[role] for role in self.data_df['role'].to_list()]

    def _check_faces(self, vals: dict):
        return vals.get("faces", "")
    
    def _load_data_into_df(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, 'r') as json_file:
            json_data = [json.loads(line) for line in json_file]

        return pd.DataFrame([{
            "sentence": vals['OCR'].lower().replace('\n', ' '),
            "original": vals['OCR'],
            "faces": self._check_faces(vals), 
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
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        item =  {
                'input_ids': torch.tensor(self.encodings.input_ids[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.encodings.attention_mask[idx], dtype=torch.long),
                'pixel_values': image,
                'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }
                
        if "deberta" in self.text_tokenizer.name_or_path.lower():
            item['token_type_ids'] = torch.tensor(self.encodings.token_type_ids[idx], dtype=torch.long)

        return item
        
    def __len__(self):
        return len(self.encoded_labels)