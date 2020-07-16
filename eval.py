import torch
import torch.nn as nn
import random

from config import MAX_LENGTH, SOS_token, DEVICE, EOS_token
from model import Encoder, Decoder
from preprocess import prepareData


def eval(lang1, lang2, reverse, enc_path, dec_path):
    input_lang, output_lang, pairs = prepareData(lang1, lang2, reverse)
    pair = random.choice(pairs)

    enc = Encoder(input_lang.n_words, hidden_size)
    dec = Decoder(output_lang.n_words, hidden_size)

    enc.load_state_dict(enc_path)
    dec.load_state_dict(dec_path)

    enc = enc.to(DEVICE)
    dec = dec.to(DEVICE)

    with torch.no_grad():
        input_tensor = input_lang.tensorFromSentence(pair[0])
        input_lenth = input_tensor.size(0)
        encoder_hidden = enc.initHidden()

        encoder_outputs = torch.zeros(max_length, enc.hidden_size, device=DEVICE)

        for ei in range(input_lenth):
            encoder_output, encoder_hidden = enc(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = dec(decoder_input, decoder_hidden)
            _, topi = decoder_output.data.topk(1)
            if topi == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.getWord(topi.item()))

            decoder_input = topi.squeeze().detach()

        return decoded_words
