import torch
from torch import nn

from nanovllm_labs.common.layers.activation import SiluAndMul
from nanovllm_labs.common.layers.linear import Linear, MergedLinear, QKVLinear
from nanovllm_labs.common.layers.rmsnorm import RMSNorm
from nanovllm_labs.common.models.qwen3 import (
    Qwen3DecoderLayerBase,
    Qwen3ForCausalLMBase,
    Qwen3ModelBase,
)

from ..layers.attention import Attention
from ..layers.embed_head import LMHead, VocabEmbedding
from ..layers.rotary_embedding import get_rope


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVLinear(
            hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
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
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(Qwen3DecoderLayerBase):
    attention_cls = Qwen3Attention
    mlp_cls = Qwen3MLP
    norm_cls = RMSNorm


class Qwen3Model(Qwen3ModelBase):
    decoder_layer_cls = Qwen3DecoderLayer
    embedding_cls = VocabEmbedding
    norm_cls = RMSNorm


class Qwen3ForCausalLM(Qwen3ForCausalLMBase):
    model_cls = Qwen3Model
    lm_head_cls = LMHead
