import torch.nn as nn


class GPT4oPromptAnswers(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return 0, batch['class_id GPT-4o']
