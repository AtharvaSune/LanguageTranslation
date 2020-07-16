import unicodedata
import string
import re
import random
from config import MAX_LENGTH, ENG_PREFIXES


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self._addWord(word)

    def _addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def getIndex(self, word):
        return self.word2index[word]

    def getWord(self, index):
        return self.index2word

    def __len__(self):
        return self.n_words


def unicodetoASCII(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodetoASCII(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLang(lang1, lang2, reverse=True):
    lines = open(f"./data/{lang1}-{lang2}.txt", encoding='utf-8').read()\
                .strip().split("\n")

    pairs = [[normalizeString(s) for s in line.split("\t")] for line in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[1].split(' ')) < MAX_LENGTH and \
        len(p[2].split(' ')) < MAX_LENGTH and \
        p[2].startswith(ENG_PREFIXES)


def filterPairs(pairs):
    return [pair[1:] for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLang(lang1, lang2, reverse)
    pairs = filterPairs(pairs)
    # print(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo reading data")
    parser.add_argument("-l1", "--lang1", type=str, required=True,
                        help="language 1 (input_language)")
    parser.add_argument("-l2", "--lang2", type=str, required=True,
                        help="language 2 (output_language)")
    parser.add_argument("-r", "--reverse", type=bool, default=False,
                        help="Reverse input/output language")
    args = parser.parse_args()

    input_lang, output_lang, pairs = prepareData(args.lang1, args.lang2,
                                                 args.reverse)
    print(random.choice(pairs))
