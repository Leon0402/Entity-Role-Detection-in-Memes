import torch
import torch.nn as nn
import transformers

import meme_entity_detection.utils.task_properties


class GPT4oPromptAnswers(nn.Module):

    def __init__(self, model_name: str = "None"):
        super().__init__()

    def forward(self, batch):
        output = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )

        return output.loss, torch.argmax(output.logits, dim=1)
