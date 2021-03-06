import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

ENG_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
