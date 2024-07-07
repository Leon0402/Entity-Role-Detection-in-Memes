import torch
import transformers
import PIL

import meme_entity_detection.utils.task_properties

from .interface import Tokenizer, Model


class DebertaTokenizer(Tokenizer):
    """
    Tokenizer for the DeBERTa model using a pre-trained DeBERTa tokenizer.
    """

    def __init__(self, tokenizer_name: str = "microsoft/deberta-v3-large"):
        """
        Initialize the DebertaTokenizer with the given model name.

        Parameters:
            model_name: Name of the pre-trained model to use.
        """
        self.processor = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize(self, texts: list[str], images: list[PIL.Image]) -> dict:
        """
        Tokenize the input texts and images.

        Parameters:
            texts: List of text strings to tokenize.
            images: List of PIL Image objects to tokenize.

        Returns:
            Dictionary containing the tokenized text and images. Refer to the documentation of the tokenizer for details.
        """
        return self.processor(texts, truncation=True, padding='max_length', return_tensors="pt", max_length=196)


class DebertaModel(Model):
    """
    Model class for Deberta for text classification.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        """
        Initialize the DeBERTa Model with the pre-trained configuration and weights.

        Parameters:
            model_name: Name of the pre-trained model to use.
        """
        super().__init__()

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=meme_entity_detection.utils.task_properties.num_classes
        )
        self.model.train()

    def forward(self, batch: dict) -> tuple[float, int]:
        """
        Forward pass of the model.

        Parameters:
            batch: Batch of input data containing input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, and labels.

        Returns:
            Tuple containing the loss and the predicted class indices.
        """
        output = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            batch['token_type_ids'],
            labels=batch['labels'],
        )

        return output.loss, torch.argmax(output.logits, dim=1)
