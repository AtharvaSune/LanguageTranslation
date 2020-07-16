import torch
import torch.optim as optim
import torch.nn as nn
from config import SOS_token, EOS_token, DEVICE, MAX_LENGTH


def trainNormal(enc, dec, input_tensor, target_tensor, optim_enc, optim_dec,
                criterion):
    encoder_hidden = enc.initHidden()
    optim_enc.zero_grad()
    optim_dec.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, enc.hidden_size, device=DEVICE)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = enc(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = dec(encoder_outputs,
                                             decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]

    loss.backward()

    optim_enc.step()
    optim_dec.step()

    return loss.item()/target_length
