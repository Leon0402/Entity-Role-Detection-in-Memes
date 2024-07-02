import torch
import torch.nn as nn
import transformers

import meme_entity_detection.utils.task_properties


class GPT4oPromptAnswers(nn.Module):

    def __init__(self, model_name: str = "None"):
        super().__init__()

    def forward(self, batch):
        return 0, batch['class_id GPT-4o']
