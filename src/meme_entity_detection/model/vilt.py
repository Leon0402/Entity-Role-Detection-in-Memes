import torch
import torch.nn as nn
import transformers

class ViltModel(nn.Module):
    def __init__(self):
        super(ViltModel, self).__init__()
        self.processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

        cfg = transformers.ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
        cfg.num_labels = 4
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
        checkpoint["embeddings.text_embeddings.position_embeddings.weight"] = torch.zeros((cfg.max_position_embeddings, 768))
        checkpoint["embeddings.text_embeddings.position_embeddings.weight"][:40] = temp

        temp = checkpoint["embeddings.token_type_embeddings.weight"]
        checkpoint["embeddings.token_type_embeddings.weight"] = torch.zeros((cfg.modality_type_vocab_size, 768))
        checkpoint["embeddings.token_type_embeddings.weight"][:3] = temp

        self.model = transformers.ViltForImagesAndTextClassification(cfg)
        self.model.vilt.load_state_dict(checkpoint, strict=True)
        self.model.train()

    def forward(self, batch):
        encoding = self.processor(
            text=batch['text'],
            images=batch['image'],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True
        )
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}

        batch_size, height, width = encoding['pixel_mask'].shape
        encoding['pixel_mask'] = encoding['pixel_mask'].view(batch_size, 1, height, width)

        output = self.model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            token_type_ids=encoding['token_type_ids'],
            pixel_values=encoding['pixel_values'],
            pixel_mask=encoding['pixel_mask'],
            labels=batch['label']
        )

        return output.loss, torch.argmax(output.logits, dim=1)