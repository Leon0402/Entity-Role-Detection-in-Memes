import torch.nn as nn
import PIL


class Tokenizer():
    """
    Interface for all tokenizers implemented for the specific models.
    """

    def tokenize(self, texts: list[str], images: list[PIL.Image]):
        pass


class Model(nn.Module):
    """
    Interface for all 
    """

    def forward(self, batch: dict) -> tuple[float, int]:
        pass
