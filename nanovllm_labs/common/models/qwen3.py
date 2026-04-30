from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen3Config


PACKED_MODULES_MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


class Qwen3DecoderLayerBase(nn.Module):
    attention_cls: type[nn.Module] | None = None
    mlp_cls: type[nn.Module] | None = None
    norm_cls: type[nn.Module] | None = None

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if self.attention_cls is None or self.mlp_cls is None or self.norm_cls is None:
            raise TypeError("Qwen3DecoderLayerBase subclasses must set attention_cls, mlp_cls, and norm_cls.")
        self.self_attn = self.attention_cls(
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
        self.mlp = self.mlp_cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = self.norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = self.norm_cls(config.hidden_size, eps=config.rms_norm_eps)

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


class Qwen3ModelBase(nn.Module):
    decoder_layer_cls: type[nn.Module] | None = None
    embedding_cls: type[nn.Module] | None = None
    norm_cls: type[nn.Module] | None = None

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if self.decoder_layer_cls is None or self.embedding_cls is None or self.norm_cls is None:
            raise TypeError("Qwen3ModelBase subclasses must set decoder_layer_cls, embedding_cls, and norm_cls.")
        self.embed_tokens = self.embedding_cls(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [self.decoder_layer_cls(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = self.norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLMBase(nn.Module):
    packed_modules_mapping = PACKED_MODULES_MAPPING
    model_cls: type[nn.Module] | None = None
    lm_head_cls: type[nn.Module] | None = None

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if self.model_cls is None or self.lm_head_cls is None:
            raise TypeError("Qwen3ForCausalLMBase subclasses must set model_cls and lm_head_cls.")
        self.model = self.model_cls(config)
        self.lm_head = self.lm_head_cls(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
