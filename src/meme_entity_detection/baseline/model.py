import torch.nn as nn
from . import config
from transformers import AutoModel


class SentimentModel(nn.Module):

    def __init__(self, label2id):
        super(SentimentModel, self).__init__()
        self.label2id = label2id
        self.num_labels = len(self.label2id)
        self.bert_layer = AutoModel.from_pretrained(config.MODEL_NAME)
        self.bert_drop = nn.Dropout(0.2)
        self.l0 = nn.Linear(1024, 256)
        self.l1 = nn.Linear(256, self.num_labels)

    def forward(self, ids, mask):
        bert_out_text = self.bert_layer(input_ids=ids, attention_mask=mask)
        dropout_out = self.bert_drop(bert_out_text['last_hidden_state'])
        output = self.l0(dropout_out)
        output = self.l1(output)
        return output
