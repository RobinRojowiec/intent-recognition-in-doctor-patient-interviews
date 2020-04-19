import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d

from utility.model_parameter import Configuration, ModelParameter


class CNNLayer(nn.Module):
    def __init__(self, config: Configuration, vocab_size=30000, use_embeddings=True, embed_dim=-1, **kwargs):
        super(CNNLayer, self).__init__()

        # set parameters
        self.max_seq_length = config.get_int(ModelParameter.MAX_LENGTH)
        self.use_gpu = torch.cuda.is_available()
        if embed_dim == -1:
            self.embedding_dim = config.get_int(ModelParameter.EMBEDDING_SIZE)
        else:
            self.embedding_dim = embed_dim
        self.max_length = config.get_int(ModelParameter.MAX_LENGTH)
        self.use_embeddings = use_embeddings
        self.conv_out_channels = config.get_int(ModelParameter.CHANNELS)
        self.filter_sizes = [2]

        # create and initialize layers
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.relu = nn.ReLU()
        self.convolutions = nn.ModuleList(
            [nn.Conv2d(1, self.conv_out_channels, (K, self.embedding_dim)) for K in self.filter_sizes])
        self.dropout = nn.Dropout(0.3)

    def get_output_length(self):
        return len(self.filter_sizes) * self.conv_out_channels

    def forward(self, samples, **kwargs):
        encoded_samples = self.encode(samples)
        return encoded_samples

    def encode(self, samples):
        x = self.embedding(samples)
        x = x.unsqueeze(1)
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convolutions]
        x = [max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = self.dropout(torch.cat(x, 1))
        return x
