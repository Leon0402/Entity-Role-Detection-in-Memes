import torch
import transformers
import PIL

import meme_entity_detection.utils.task_properties

from .interface import Tokenizer, Model


class DebertaTokenizer(Tokenizer):

    def __init__(self, tokenizer_name: str = "microsoft/deberta-v3-large"):
        self.processor = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize(self, texts: list[str], images: list[PIL.Image]) -> dict:
        return self.processor(texts, truncation=True, padding='max_length', return_tensors="pt", max_length=196)


class DebertaModel(Model):

    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        super().__init__()

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=meme_entity_detection.utils.task_properties.num_classes
        )
        self.model.train()

    def forward(self, batch):
        output = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            batch['token_type_ids'],
            labels=batch['labels'],
        )

        return output.loss, torch.argmax(output.logits, dim=1)
