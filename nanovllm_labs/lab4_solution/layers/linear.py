import torch
from torch import nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedLinear(Linear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        param.data[shard_offset : shard_offset + shard_size].copy_(loaded_weight)


class QKVLinear(Linear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        q_size = self.total_num_heads * self.head_size
        kv_size = self.total_num_kv_heads * self.head_size
        if loaded_shard_id == "q":
            offset = 0
            size = q_size
        elif loaded_shard_id == "k":
            offset = q_size
            size = kv_size
        else:
            offset = q_size + kv_size
            size = kv_size
        param.data[offset : offset + size].copy_(loaded_weight)
