# implementation of Bidirectional transformers

from torch.nn import Module
import torch.nn
from transformers import BertTokenizer

class BERT(Module):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # using a pretrained tokenizer because I don't wanna do it myself but I do want the same one as actual BERT
