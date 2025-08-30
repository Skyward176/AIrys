# Implementation of a dense query retrieval component to be used for my RAG architecture
# Implemented based on study of the 30th of September 2020 Paper "Dense Passage Retrieval for Open-Domain Question Answering"
# https://arxiv.org/abs/2005.11401

# is this not simply just an embedding layer trained in a particular way?? Probably yes, I'll need to read more carefully. Because if it is this is great news.
from torch.nn import Module
from torch import nn

class DenseQueryRetrieval(Module):
    def __init__(self, input_dim, output_dim):
        self.embedding = nn.Embedding(input_dim, output_dim) # we initialize an embedding layer