import pickle

import torch
import torch.nn as nn
from torchtext.data import Field

from common.paths import ROOT_RELATIVE_DIR, MODEL_PATH
from models.bert_layer import BERTLayer
from probability.tables import TransitionTable
from utility.model_parameter import Configuration, ModelParameter


class BERTWithConversationContext(nn.Module):
    def __init__(self, config: Configuration, class_count=100,
                 hidden_dim=100,
                 class_field=None, device=torch.device('cpu')):
        """
        Simple embedding using position and previous class
        :param config:
        :param class_count:
        :param vocab_size:
        """
        super(BERTWithConversationContext, self).__init__()

        # set parameters
        self.hidden_dim = hidden_dim
        self.class_field: Field = class_field
        self.class_count = class_count
        self.max_length = config.get_int(ModelParameter.MAX_LENGTH)
        self.embedding_size = config.get_int(ModelParameter.EMBEDDING_SIZE)
        self.device = device

        # configure history components
        self.with_position_embedding = False
        self.with_class_embedding = True
        self.with_utterance_classifier = False
        self.with_transition_probs = False

        # create and initialize layers
        # learns embedding vector for class labels
        self.class_embedding = nn.Embedding(self.class_count, self.embedding_size)

        # learns class label for positions
        self.position_embedding = nn.Embedding(100, self.embedding_size)

        # load probability table and neural_bert_models
        with open(ROOT_RELATIVE_DIR + MODEL_PATH + "transition_table.pckl", "rb") as file_prt:
            transition_table: TransitionTable = pickle.load(file_prt)
            transition_table.lambda_value = 1
            transition_table.class_field = class_field

            fixed_embedding = transition_table.create_probability_matrix(device=self.device)
            self.transition_embedding = nn.Embedding(self.class_count, self.class_count)
            self.transition_embedding.weight.data.copy_(fixed_embedding)
            self.transition_embedding.weight.requires_grad = False

        # embed previous sample
        self.utterance_classifier = BERTLayer(device=self.device)

        # output layer
        self.dropout = nn.Dropout(p=0.5)
        self.linear_input_size = self.get_embeddings_length() + self.get_additional_length() + self.utterance_classifier.get_output_length()
        self.linear_layer = nn.Linear(self.linear_input_size, class_count)
        self.softmax = nn.Softmax(dim=0)

    def get_embeddings_length(self):
        multiplier = 0
        if self.with_position_embedding:
            multiplier += 1
        if self.with_class_embedding:
            multiplier += 1
        return self.embedding_size * multiplier

    def get_additional_length(self):
        add_length = 0
        if self.with_utterance_classifier:
            add_length += self.utterance_classifier.get_output_length()
        if self.with_transition_probs:
            add_length += self.class_count
        return add_length

    def forward(self, sample, previous_classes, positions, previous_sample, *args, **kwargs):
        representations = []

        # embed sample
        sample_embed = self.utterance_classifier(sample)
        representations.append(sample_embed)

        # encode the previous utterance into a matrix
        if self.with_utterance_classifier:
            with torch.no_grad():
                self.utterance_classifier.eval()
                representations.append(self.utterance_classifier(previous_sample))

        # learn vector representations for position and class
        if self.with_class_embedding:
            embed_previous_classes = self.class_embedding(previous_classes)
            representations.append(embed_previous_classes)

        if self.with_position_embedding:
            embed_position = self.position_embedding(positions)
            representations.append(embed_position)

        # transition probs and apply dropout and
        if self.with_transition_probs:
            representations.append(self.transition_embedding(previous_classes))

        # concatenate all representations
        concat_class_and_position = torch.cat(representations, 1)
        concat_class_and_position = self.dropout(concat_class_and_position)

        # linear transformation
        output = self.linear_layer(concat_class_and_position)

        # calculate probabilities
        probs = self.softmax(output)

        return output, probs
