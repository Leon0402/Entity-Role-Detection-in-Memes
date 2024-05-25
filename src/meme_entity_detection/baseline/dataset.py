from . import config
import torch


class AspectRole:

    def __init__(self, tokenizer, data, label2id):
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.text_encodings = self.tokenizer(data['sentence'].to_list(),
                                             data['word'].to_list(),
                                             truncation=True,
                                             padding='max_length',
                                             max_length=config.MAX_LEN)
        self.encoded_labels = [
            label2id[role] for role in data['role'].to_list()
        ]

    def __getitem__(self, idx):
        return {
            'input_ids':
            torch.tensor(self.text_encodings.input_ids[idx], dtype=torch.long),
            'attention_mask':
            torch.tensor(self.text_encodings.attention_mask[idx],
                         dtype=torch.long),
            'token_type_ids':
            torch.tensor(self.text_encodings.token_type_ids[idx],
                         dtype=torch.long),
            'labels':
            torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.encoded_labels)
