import torch
from torch import nn
from transformers import Qwen3Config

from ..layers.activation import SiluAndMul
from ..layers.attention import Attention
from ..layers.embed_head import LMHead, VocabEmbedding
from ..layers.layernorm import RMSNorm
from ..layers.linear import Linear, MergedLinear, QKVLinear
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        *,
        start_layer: int,
        end_layer: int,
        is_first_stage: bool,
        is_last_stage: bool,
    ) -> None:
        super().__init__()
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        self.layer_ids = list(range(start_layer, end_layer))
        if is_first_stage:
            self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.register_module("embed_tokens", None)

        self.layers = nn.ModuleDict(
            {str(layer_id): Qwen3DecoderLayer(config) for layer_id in self.layer_ids}
        )

        if is_last_stage:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.register_module("norm", None)

    def forward(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_first_stage:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert hidden_states is not None
            assert residual is not None

        for layer_id in self.layer_ids:
            hidden_states, residual = self.layers[str(layer_id)](positions, hidden_states, residual)

        if self.is_last_stage:
            hidden_states, _ = self.norm(hidden_states, residual)
            residual = hidden_states

        return hidden_states, residual


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        *,
        start_layer: int,
        end_layer: int,
        is_first_stage: bool,
        is_last_stage: bool,
    ) -> None:
        super().__init__()
        self.is_last_stage = is_last_stage
        self.model = Qwen3Model(
            config,
            start_layer=start_layer,
            end_layer=end_layer,
            is_first_stage=is_first_stage,
            is_last_stage=is_last_stage,
        )
        if is_last_stage:
            self.lm_head = LMHead(config.vocab_size, config.hidden_size)
            if config.tie_word_embeddings and is_first_stage:
                self.lm_head.weight.data = self.model.embed_tokens.weight.data
        else:
            self.register_module("lm_head", None)

    def forward(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            positions,
            input_ids=input_ids,
            hidden_states=hidden_states,
            residual=residual,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self.lm_head is not None
        return self.lm_head(hidden_states)
