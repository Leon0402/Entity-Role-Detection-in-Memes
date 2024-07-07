import torch
import transformers
import PIL

import meme_entity_detection.utils.task_properties

from .interface import Tokenizer, Model


class ViltTokenizer(Tokenizer):

    def __init__(self, model_name: str = "FacebookAI/roberta-large"):
        self.processor = transformers.ViltProcessor.from_pretrained(model_name)

    def tokenize(self, texts: list[str], images: list[PIL.Image]) -> dict:
        encoding = self.processor(
            text=texts, images=images, return_tensors="pt", padding="max_length", max_length=196, truncation=True
        )
        batch_size, height, width = encoding['pixel_mask'].shape
        encoding['pixel_mask'] = encoding['pixel_mask'].view(batch_size, 1, height, width)
        return encoding


class ViltModel(Model):

    def __init__(self):
        super().__init__()

        cfg = transformers.ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
        cfg.num_labels = meme_entity_detection.utils.task_properties.num_classes
        cfg.type_vocab_size = 5
        cfg.max_position_embeddings = 275
        cfg.num_images = 1
        cfg.modality_type_vocab_size = cfg.modality_type_vocab_size + cfg.num_images
        cfg.merge_with_attentions = True

        checkpoint = transformers.ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2").state_dict()

        # Correct some weights because some parameters changed
        temp = checkpoint["embeddings.text_embeddings.token_type_embeddings.weight"]
        checkpoint["embeddings.text_embeddings.token_type_embeddings.weight"] = torch.zeros((cfg.type_vocab_size, 768))
        checkpoint["embeddings.text_embeddings.token_type_embeddings.weight"][:2, :] = temp

        temp = checkpoint["embeddings.text_embeddings.position_embeddings.weight"]
        checkpoint["embeddings.text_embeddings.position_embeddings.weight"] = torch.zeros(
            (cfg.max_position_embeddings, 768)
        )
        checkpoint["embeddings.text_embeddings.position_embeddings.weight"][:40] = temp

        temp = checkpoint["embeddings.token_type_embeddings.weight"]
        checkpoint["embeddings.token_type_embeddings.weight"] = torch.zeros((cfg.modality_type_vocab_size, 768))
        checkpoint["embeddings.token_type_embeddings.weight"][:3] = temp

        self.model = transformers.ViltForImagesAndTextClassification(cfg)
        self.model.vilt.load_state_dict(checkpoint, strict=True)
        self.model.train()

    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            pixel_values=batch['pixel_values'],
            pixel_mask=batch['pixel_mask'],
            labels=batch['labels'],
        )

        return output.loss, torch.argmax(output.logits, dim=1)
