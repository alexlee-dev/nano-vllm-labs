from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class VocabEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class LMHead(VocabEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        context_getter: Callable[[], object],
        bias: bool = False,
    ) -> None:
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self.context_getter = context_getter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.context_getter()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return F.linear(x, self.weight)
