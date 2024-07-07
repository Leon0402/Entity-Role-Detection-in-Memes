import torch
import torch.nn as nn
import transformers
import PIL

import meme_entity_detection.utils.task_properties

from .interface import Tokenizer, Model


class RobertaTokenizer(Tokenizer):

    def __init__(self, model_name: str = "FacebookAI/roberta-large"):
        self.processor = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(self, texts: list[str], images: list[PIL.Image]) -> dict:
        return self.processor(texts, truncation=True, padding='max_length', return_tensors="pt", max_length=196)


class RobertaModel(Model):

    def __init__(self, model_name: str = "FacebookAI/roberta-large"):
        super().__init__()

        self.config = transformers.AutoConfig.from_pretrained(model_name)
        self.config.num_labels = meme_entity_detection.utils.task_properties.num_classes
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)

        # TODO: Linear Probing might be an option too -> Seems more difficult initially
        self.model.train()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        self.loss_fn = nn.CrossEntropyLoss()
        # Weights calculated with sklearn for non balanced case, but seems to perform worse in comarison to dataset balancing
        # weight=torch.tensor([0.42902496, 2.02010309, 1.32787441, 2.37515152]) # Balanced
        # weight=torch.tensor([0.31955189, 4.81153846, 1.80407911, 9.21789474])) # Unbalanced

    def forward(self, batch):
        output = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            # labels=batch['labels'],
        )
        output.loss = self.loss_fn(output.logits.view(-1, self.config.num_labels), batch['labels'].view(-1))

        return output.loss, torch.argmax(output.logits, dim=1)
