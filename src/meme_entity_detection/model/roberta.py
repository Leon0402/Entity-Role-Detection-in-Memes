import torch
import torch.nn as nn
import transformers

import meme_entity_detection.utils.task_properties


class RobertaModel(nn.Module):

    def __init__(self, model_name: str = "FacebookAI/roberta-large"):
        super().__init__()

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=meme_entity_detection.utils.task_properties.num_classes
        )
        self.model.train()

    def forward(self, batch):
        output = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )

        return output.loss, torch.argmax(output.logits, dim=1)
