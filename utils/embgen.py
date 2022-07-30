import torch
import torch as th
import torch.nn as nn


class Embedder(nn.Module):
    """ Encodes the given text input into a high dimensional embedding vector
        uses LSTM internally
    """

    def __init__(self, embedding_size, hidden_size, num_layers, device=th.device("cpu")):
        """
        constructor of the class
        :param embedding_size: size of the input embeddings
        :param vocab_size: size of the vocabulary
        :param hidden_size: hidden size of the LSTM network
        :param num_layers: number of LSTM layers in the network
        :param device: device on which to run the Module
        """
        super(Embedder, self).__init__()

        # create the state:
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create the LSTM layer:
        from torch.nn import Sequential, LSTM
        self.network = Sequential(
            LSTM(self.embedding_size, self.hidden_size,
                 self.num_layers, batch_first=True)
        ).to(device)

    def forward(self, x):
        """
        performs forward pass on the given data:
        :param x: input numeric sequence
        :return: enc_emb: encoded text embedding
        """

        output, (_, _) = self.network(x)
        return output
