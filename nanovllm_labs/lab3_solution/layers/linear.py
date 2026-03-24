from __future__ import annotations


from torch import nn
import torch
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    
class MergedLinear(Linear):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int) -> None:
        offset = sum(self.output_sizes[:loaded_shard_id])
        size = self.output_sizes[loaded_shard_id]
        param.data[offset : offset + size].copy_(loaded_weight)

class QKVLinear(Linear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        output_size = (num_heads + 2 * num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str) -> None:
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if loaded_shard_id == "q":
            offset = 0
            size = q_size
        elif loaded_shard_id == "k":
            offset = q_size
            size = kv_size
        elif loaded_shard_id == "v":
            offset = q_size + kv_size
            size = kv_size
        param.data[offset : offset + size].copy_(loaded_weight)
