import torch
import torch.nn as nn
import math


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.types.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)
