import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import DEVICE, MAX_LENGTH


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


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p,
                 max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, h, encoder_out):
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], h[0]), 1)),
            dim=1
        )

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_out.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights
