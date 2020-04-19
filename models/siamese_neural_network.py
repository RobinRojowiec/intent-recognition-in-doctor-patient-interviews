import random

import torch
import torch.nn as nn

from models.cnn_layer import CNNLayer
from utility.model_parameter import ModelParameter, Configuration


class SiameseNeuralNetwork(nn.Module):
    def __init__(self, config: Configuration, label_count=64, device=torch.device('cpu'), *args, **kwargs):
        super(SiameseNeuralNetwork, self).__init__()

        # set parameters
        self.max_length = config.get_int(ModelParameter.MAX_LENGTH)
        self.device = device

        # create and initialize layers
        self.cnn_layer = CNNLayer(config)
        self.distance_measure = nn.CosineSimilarity()

    def distance(self, a, b):
        return self.distance_measure(a, b)

    def get_output_dim(self):
        return self.cnn_layer.get_output_length()

    def forward(self, sample, previous_classes, positions, previous_sample, sample_pos, *sample_neg, **kwargs):
        n_negative = len(sample_neg)
        selected_negative = sample_neg[random.randint(0, n_negative - 1)]
        return self.compare(sample, sample_pos, mode=kwargs["mode"]), self.compare(sample, selected_negative,
                                                                                   mode=kwargs["mode"])

    def get_features(self, sample):
        return self.cnn_layer(sample)

    def compare(self, sample_1, sample_2, mode="train", **kwargs):
        encoded_sample_1 = self.cnn_layer(sample_1)
        encoded_sample_2 = self.cnn_layer(sample_2)

        return self.distance(encoded_sample_1, encoded_sample_2)
