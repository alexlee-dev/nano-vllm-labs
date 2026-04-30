import torch
from torch import nn

from nanovllm_labs.common.layers.activation import SiluAndMul
from nanovllm_labs.common.models.qwen3 import (
    Qwen3DecoderLayerBase,
    Qwen3ForCausalLMBase,
    Qwen3ModelBase,
)
from nanovllm_labs.lab2_solution.layers.attention import Attention
from nanovllm_labs.lab2_solution.layers.embed_head import Embedding, LMHead
from nanovllm_labs.lab2_solution.layers.linear import Linear, MergedLinear, QKVLinear
from nanovllm_labs.lab2_solution.layers.rmsNorm import RMSNorm
from nanovllm_labs.lab2_solution.layers.rotary_embedding import get_rope


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        head_dim: int | None = None,
        rope_theta: float = 1000000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.qkv_proj = QKVLinear(
            hidden_size,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = Linear(self.q_size, hidden_size, bias=qkv_bias)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*hidden_states.shape[:-1], self.num_kv_heads, self.head_dim)
        v = v.view(*hidden_states.shape[:-1], self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v).reshape(*hidden_states.shape[:-1], -1)
        return self.o_proj(attn_output)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedLinear(hidden_size, [intermediate_size, intermediate_size], bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu", f"Unsupported activation: {hidden_act}"
        self.activation_func = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.activation_func(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(Qwen3DecoderLayerBase):
    attention_cls = Qwen3Attention
    mlp_cls = Qwen3MLP
    norm_cls = RMSNorm


class Qwen3Model(Qwen3ModelBase):
    decoder_layer_cls = Qwen3DecoderLayer
    embedding_cls = Embedding
    norm_cls = RMSNorm


class Qwen3ForCausalLM(Qwen3ForCausalLMBase):
    model_cls = Qwen3Model
    lm_head_cls = LMHead
