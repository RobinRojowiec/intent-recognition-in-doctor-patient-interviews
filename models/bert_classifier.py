import torch
import torch.nn as nn

from models.bert_layer import BERTLayer
from utility.model_parameter import Configuration, ModelParameter


class BERTClassifier(nn.Module):
    def __init__(self, config: Configuration, num_labels, device=torch.device('cpu'), **kwargs):
        super(BERTClassifier, self).__init__()

        self.device = device
        self.max_length = config.get_int(ModelParameter.MAX_LENGTH)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(768, num_labels)
        self.softmax = nn.Softmax(dim=1)

        self.bert_layer = BERTLayer(device=self.device)

    @staticmethod
    def attn_value(x):
        """
        Returns the value for attention matrix using zero for pad tokens
        :param x:
        :return:
        """
        if x > 0:
            return 1
        return 0

    def forward(self, sample, previous_classes, positions, previous_sample, *args, **kwargs):
        # produce pooled output of the last encoder
        q_pooled = self.bert_layer(sample)

        # apply dropout and linear projection
        logits = self.linear(self.dropout(q_pooled))
        return logits, self.softmax(logits)
