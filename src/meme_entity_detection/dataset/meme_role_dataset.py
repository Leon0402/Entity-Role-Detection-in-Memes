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


class MemeRoleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_path: Path, 
        balance_dataset: bool = False, 
        use_faces=False,
        ocr_type="OCR", # use GPT-4o or OCR
        use_gpt_description: bool = False):
        assert ocr_type=="OCR" or ocr_type=="GPT-4o"
        
        self.data_df = self._load_data_into_df(file_path)
        self.use_faces = use_faces
        self.ocr_type = ocr_type
        self.use_gpt_description = use_gpt_description


        self.image_base_dir = file_path.parent.parent / "images"

        if balance_dataset:
            self.data_df = self._balance_dataset(self.data_df)

        label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}
        self.encoded_labels = [label2id[role] for role in self.data_df['role'].to_list()]

        self.processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

    def _load_data_into_df(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, 'r') as json_file:
            json_data = [json.loads(line) for line in json_file]

        all_roles = ['hero', 'villain', 'victim', 'other']   
        df =  pd.DataFrame([{
            "sentence": vals['OCR'].lower().replace('\n', ' '),
            "sentence GPT-4o": (vals["OCR GPT-4o"] if vals["OCR GPT-4o"] else "").lower().replace('\n', ' '),
            "description GPT-4o": (vals["IMAGE DESCRIPTION GPT-4o"] if vals["IMAGE DESCRIPTION GPT-4o"] else "").lower().replace('\n', ' '),
            "classification GPT-4o": defaultdict(lambda: "other", vals["CLASSIFICATION GPT-4o"]) if vals["CLASSIFICATION GPT-4o"] else defaultdict(lambda: "other"),
            "original": vals['OCR'],
            "faces": vals.get("faces", ""),
            "word": word_val,
            "image": vals['image'],
            "role": role
        } for vals in tqdm(json_data) for role in all_roles for word_val in vals[role]])



        df["classification GPT-4o"]  = [data["classification GPT-4o"][data["word"]] for _, data in tqdm(df[["classification GPT-4o", "word"]].iterrows())]
        df["classification GPT-4o"] = df["classification GPT-4o"].str.replace("villian", "villain")
        df["classification GPT-4o"] = df["classification GPT-4o"].apply(lambda x: self._correct_gpt4o_classification(x, all_roles))
        df["class_id GPT-4o"] = df["classification GPT-4o"].apply(lambda x: {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}[x])
        

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
        faces = " - ".join(row["faces"] or []) if self.use_faces else ""
        
        if self.ocr_type == "GPT-4o" and row["sentence GPT-4o"] != "": #use only gpt ocr if enabled and available
            row["sentence"] = row["sentence GPT-4o"]
        
        description = ""
        if self.use_gpt_description:
            description += " [SEP] " + row["description GPT-4o"]
        
        # TODO: Check how to construct text here
        return {
            "image": image,
            "text": row["sentence"] + " [SEP] " + row["word"] + " [SEP] " + faces + description,
            "class_id GPT-4o": row["class_id GPT-4o"],
            "label": torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.encoded_labels)

    @staticmethod
    def collate_fn(batch, processor):
        gpt4_classifications = torch.tensor([item['class_id GPT-4o'] for item in batch])
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        # TODO: Check what value makes sense for max_length. One of the paper uses 275, but this significantly slowed down training
        encoding = processor(
            text=texts, images=images, return_tensors="pt", padding="max_length", max_length=64, truncation=True
        )
        batch_size, height, width = encoding['pixel_mask'].shape
        encoding['pixel_mask'] = encoding['pixel_mask'].view(batch_size, 1, height, width)

        encoding['labels'] = labels
        encoding['class_id GPT-4o'] = gpt4_classifications

        return encoding
