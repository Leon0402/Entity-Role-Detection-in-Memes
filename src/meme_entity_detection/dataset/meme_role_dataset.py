from pathlib import Path
import json
import torch
import transformers
import torchvision.transforms
import pandas as pd
import sklearn.utils
from tqdm import tqdm
from PIL import Image

class MemeRoleDataset(torch.utils.data.Dataset):

    def __init__(self, file_path: Path, balance_dataset: bool = False, text_tokenizer: str = "microsoft/deberta-v3-large", image_tokenizer: str = "google/vit-base-patch16-224", use_faces=False):
        self.data_df = self._load_data_into_df(file_path)
        self.use_faces = use_faces

        self.image_base_dir = file_path.parent.parent / "images"

        if balance_dataset:
            self.data_df = self._balance_dataset(self.data_df)

        self.tokenizer = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

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
        image = self.image_transform(image)
        
        encoding = self.tokenizer(
            text=self.sentences[idx],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True
        )
        
        item = {
                'input_ids': encoding.input_ids.squeeze(0),
                'token_type_ids': encoding.token_type_ids.squeeze(0),
                'attention_mask': encoding.attention_mask.squeeze(0),
                'pixel_values': encoding.pixel_values.squeeze(0),
                'pixel_mask': encoding.pixel_mask,
                'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

        return item
        
    def __len__(self):
        return len(self.encoded_labels)
