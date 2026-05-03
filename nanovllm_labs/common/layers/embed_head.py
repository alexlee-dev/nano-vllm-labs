import torch
import torch.nn.functional as F
from torch import nn


class VocabEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class LMHead(VocabEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, get_context, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self._get_context = get_context

    def forward(self, x: torch.Tensor):
        context = self._get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return F.linear(x, self.weight)
