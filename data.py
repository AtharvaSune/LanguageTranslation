import torch
from torch.utils.data import Dataset
from config import EOS_token, DEVICE


class CustomDataSet():
    def __init__(self, lang1, lang2, pairs):
        self.lang1 = lang1
        self.lang2 = lang2
        self.pairs = pairs
        self.len = len(pairs)

    def __len__(self):
        return self.len

    def indexFromSentence(self, lang, sentence):
        return [lang.getIndex(word) for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=DEVICE)\
            .view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(self.lang1, pair[0])
        output_tensor = self.tensorFromSentence(self.lang2, pair[1])

        return input_tensor, output_tensor


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser("description = test dataset class")
    parser.add_argument("-l1", "--lang1", required=True, type=str,
                        help="input_language")
    parser.add_argument("-l2", "--lang2", required=True, type=str,
                        help="output_language")
    parser.add_argument("-r", "--reverse", type=bool, default=False)
    args = parser.parse_args()

    from preprocess import prepareData
    input_lang, output_lang, pairs = prepareData(args.lang1, args.lang2,
                                                 args.reverse)

    rawdataset = CustomDataSet(input_lang, output_lang, pairs)
    print(rawdataset.__getitem__(random.choice(range(len(rawdataset)))))
