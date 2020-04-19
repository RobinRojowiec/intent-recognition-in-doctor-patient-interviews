import os

import torch
import torch.nn as nn
from pytorch_transformers import BertModel

from common.paths import ROOT_RELATIVE_DIR, MODEL_PATH


class BERTLayer(nn.Module):
    def __init__(self, device=torch.device('cpu'), **kwargs):
        super(BERTLayer, self).__init__()

        self.bert_model_path = 'bert-base-german-cased'
        file_name = ROOT_RELATIVE_DIR + MODEL_PATH + self.bert_model_path + ".model"
        if not os.path.exists(file_name):
            self.bert_model = BertModel.from_pretrained(self.bert_model_path)
        else:
            self.bert_model = BertModel.from_pretrained(file_name)
        self.device = device

    @staticmethod
    def get_output_length():
        return 768

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

    def forward(self, sample, *args, **kwargs):
        # generate segment type ids for sentence (only one segment)
        bert_question_type_ids = torch.LongTensor(
            [[0 for _ in range(sample.size()[1])] for _ in range(sample.size()[0])])
        bert_question_type_ids = bert_question_type_ids.to(self.device)

        # generate attention mask to padding tokens
        bert_attn_mask = torch.LongTensor(
            [[self.attn_value(x) for x in range(sample.size()[1])] for _ in
             range(sample.size()[0])])
        bert_attn_mask = bert_attn_mask.to(self.device)

        # produce pooled output of the last encoder
        outputs = self.bert_model(input_ids=sample, token_type_ids=bert_question_type_ids,
                                  attention_mask=bert_attn_mask)

        # apply pooling
        return torch.max(outputs[0], 1)[0]
