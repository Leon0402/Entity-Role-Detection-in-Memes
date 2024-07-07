import torch.nn as nn

from .interface import Model


class GPT4oPromptAnswers(Model):
    """
    Model class for GPT-4o for text classification.
    """

    def forward(self, batch: dict) -> tuple[float, int]:
        """
        Forward pass of the model. This will just return the predictions from ChatGPT-4o that have been added to the dataset in preprocessing.

        Parameters:
            batch: Batch of input data containing input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, and labels.

        Returns:
            Tuple containing the loss and the predicted class indices.
        """
        return 0, batch['class_id GPT-4o']
