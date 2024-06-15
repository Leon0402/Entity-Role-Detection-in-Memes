from pathlib import Path
import json
import torch
import transformers
import pandas as pd
import sklearn.utils
from tqdm import tqdm

class MemeRoleDataset(torch.utils.data.Dataset):

    def __init__(self, file_path: Path, balance_dataset: bool = False, tokenizer: str = "microsoft/deberta-v3-large", use_faces=False):
        self.data_df = self._load_data_into_df(file_path)
        self.use_faces = use_faces
        # TODO: balancing logic seems slightly weird, maybe double check, but it was used like this in the original code
        if balance_dataset:
            self.data_df = self._balance_dataset(self.data_df)

        if "deberta" in tokenizer.lower():
            self.tokenizer_type = "deberta"
        elif "roberta" in tokenizer.lower():
            self.tokenizer_type = "roberta"
                                  
        sentences = self.data_df['sentence'] 
        if self.use_faces:# add faces to the sentences
            sentences = sentences + " [SEP] - "+ self.data_df["faces"].apply(lambda x: x[0] if x else "")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.encodings = tokenizer(
            sentences.to_list(),
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
            "faces": self._check_faces(vals), #only check for faces if faces are part of the dataset
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
        item =  {
                'input_ids': torch.tensor(self.encodings.input_ids[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.encodings.attention_mask[idx], dtype=torch.long),
                'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }
                
        if self.tokenizer_type == "deberta":
            item['token_type_ids'] = torch.tensor(self.encodings.token_type_ids[idx], dtype=torch.long),

        return item
        
    def __len__(self):
        return len(self.encoded_labels)
