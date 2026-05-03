import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm_labs.common.embed_head import LMHead, VocabEmbedding
from nanovllm_labs.common.layernorm import RMSNorm
from nanovllm_labs.common.qwen3_blocks import Qwen3DecoderLayer
from ..utils.context import get_context


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
            {
                str(layer_id): Qwen3DecoderLayer(config, context_getter=get_context)
                for layer_id in self.layer_ids
            }
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
            self.lm_head = LMHead(
                config.vocab_size,
                config.hidden_size,
                context_getter=get_context,
            )
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
