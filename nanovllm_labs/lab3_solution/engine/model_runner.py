import os

import torch
from transformers import AutoConfig

from nanovllm_labs.lab3_solution.engine.sequence import Sequence
from nanovllm_labs.lab3_solution.layers.sampler import Sampler
from nanovllm_labs.lab3_solution.models.qwen3 import Qwen3ForCausalLM
from nanovllm_labs.lab3_solution.utils.context import reset_context, set_context
from nanovllm_labs.lab3_solution.utils.loader import load_model


class ModelRunner:
    def __init__(
        self,
        model: str,
        block_size: int,
        gpu_memory_utilization: float,
        dtype: str = "auto",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        self.model_path = os.path.expanduser(model)
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization

        torch.cuda.set_device(0)
        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if dtype == "auto":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype={dtype!r}")
        self.model_dtype = torch_dtype
        self.device = torch.device("cuda", 0)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(self.hf_config)
        load_model(self.model, self.model_path)
        self.sampler = Sampler()
        self.allocate_kv_cache()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def allocate_kv_cache(self) -> None:
        free, _ = torch.cuda.mem_get_info()
        num_kv_heads = self.hf_config.num_key_value_heads
        head_dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        block_bytes = (
            2
            * self.hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=self.model_dtype).element_size()
        )
        self.num_kvcache_blocks = max(1, int(free * self.gpu_memory_utilization) // block_bytes)
        self.kv_cache = torch.empty(
            2,
            self.hf_config.num_hidden_layers,
            self.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=self.model_dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            uncached = seq[seq.num_cached_tokens :]
            input_ids.extend(uncached)
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen)
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                end = start + (self.block_size if i != seq.num_blocks - 1 else seq.last_block_num_tokens)
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_k=cu_seqlens_k_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_t,
            block_tables=block_tables,
        )
        return input_ids_t, positions_t
    
    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = [seq.last_token for seq in seqs]
        positions = [len(seq) - 1 for seq in seqs]
        slot_mapping = [seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1 for seq in seqs]
        context_lens = [len(seq) for seq in seqs]
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables_t = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping_t, context_lens=context_lens_t, block_tables=block_tables_t)
        return input_ids_t, positions_t

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.model.compute_logits(self.model(input_ids, positions))
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids

    def exit(self) -> None:
        del self.model, self.kv_cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
