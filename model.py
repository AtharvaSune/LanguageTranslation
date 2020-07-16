import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import DEVICE


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, h):
        embedded = self.embedding(x).view(1, 1, -1)
        out = embedded
        out, h = self.gru(out, h)
        return out, h

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h):
        out = self.embedding(x).view(1, 1, -1)
        out = F.relu(out)
        out, h = self.gru(out, h)
        out = self.softmax(self.linear(out[0]))

        return out, h

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
