from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm_labs.lab3_solution.utils.context import get_context


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
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlen_q[1:] - 1
            x = x[last_indices].contiguous()
        return F.linear(x, self.weight)
