from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token ids [batch, seq] are looked up in the table [vocab, hidden] to produce embeddings [batch, seq, hidden].
        return F.embedding(x, self.weight)


class LMHead(Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False) -> None:
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden states [batch, seq, hidden] are projected with [vocab, hidden] to logits [batch, seq, vocab].
        return F.linear(x, self.weight)
