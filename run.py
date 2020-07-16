import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import pickle as pkl

from model import Encoder, Decoder
from train import trainNormal
from data import CustomDataSet
from preprocess import prepareData
from config import SOS_token, EOS_token, DEVICE


def main(lang1, lang2, reverse, batch_size, n_epochs, hidden_size, plot_every):
    input_lang, output_lang, pairs = prepareData(lang1, lang2, reverse)

    enc = Encoder(input_lang.n_words, hidden_size).to(DEVICE)
    dec = Decoder(output_lang.n_words, hidden_size).to(DEVICE)
    optim_enc = optim.Adam(enc.parameters())
    optim_dec = optim.Adam(dec.parameters())
    criterion = nn.NLLLoss()
    plot_losses = []

    for ep in range(n_epochs):
        total_loss = 0
        training_pairs = random.choices(pairs, k=batch_size)

        for i in range(len(training_pairs)):
            training_pairs[i][0] = input_lang.tensorFromSentence(
                                            training_pairs[i][0])
            training_pairs[i][1] = output_lang.tensorFromSentence(
                                            training_pairs[i][1])

        for batch, (input_tensor, target_tensor) in enumerate(training_pairs):
            loss = trainNormal(enc, dec, input_tensor, target_tensor,
                               optim_enc, optim_dec, criterion)
            total_loss += loss

        if (ep+1) % plot_every == 0:
            plot_losses.append(total_loss/plot_every)
            print(f"ep: {ep+1} loss: {total_loss/plot_every}")

    torch.save(enc.state_dict(), "encoder.pth")
    torch.save(dec.state_dict(), "decoder.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l1", "--lang1", required=True, type=str,
                        help="input_language")
    parser.add_argument("-l2", "--lang2", required=True, type=str,
                        help="output_language")
    parser.add_argument("-r", "--reverse", type=bool, default=False)
    parser.add_argument("-b", "--batch", required=True, type=int,
                        help="batch size")
    parser.add_argument("-ep", "--epochs", required=True, type=int,
                        help="number of epochs")
    parser.add_argument("-hi", "--hidden_size", required=True, type=int)
    parser.add_argument("-p", "--plot", required=True, type=int)
    args = parser.parse_args()

    main(args.lang1, args.lang2, args.reverse, args.batch,
         args.epochs, args.hidden_size, args.plot)
