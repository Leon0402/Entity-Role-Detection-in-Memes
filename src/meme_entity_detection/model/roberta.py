import torch
import torch.nn as nn
import transformers
import PIL

import meme_entity_detection.utils.task_properties

from .interface import Tokenizer, Model


class RobertaTokenizer(Tokenizer):
    """
    Tokenizer for the RoBERTa model using a pre-trained RoBERTa tokenizer.
    """

    def __init__(self, model_name: str = "FacebookAI/roberta-large"):
        """
        Initialize the RobertaTokenizer with the given model name.

        Parameters:
            model_name: Name of the pre-trained model to use.
        """
        self.processor = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)

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


class RobertaModel(Model):
    """
    Model class for RoBERTa for text classification.
    """

    def __init__(self, model_name: str = "FacebookAI/roberta-large"):
        """
        Initialize the RoBERTa Model with the pre-trained configuration and weights.

        Parameters:
            model_name: Name of the pre-trained model to use.
        """
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
            # labels=batch['labels'],
        )
        output.loss = self.loss_fn(output.logits.view(-1, self.config.num_labels), batch['labels'].view(-1))

        return output.loss, torch.argmax(output.logits, dim=1)
