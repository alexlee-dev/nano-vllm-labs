from __future__ import annotations

import os

import torch
from transformers import AutoConfig

from nanovllm_labs.common.loader import load_model
from nanovllm_labs.common.sequence import Sequence
from nanovllm_labs.lab4_solution.utils.context import get_context
from nanovllm_labs.lab4_solution.layers.sampler import Sampler
from nanovllm_labs.lab4_solution.models.qwen3 import Qwen3ForCausalLM
from nanovllm_labs.lab4_solution.utils.context import reset_context, set_context
from nanovllm_labs.sampling_params import SamplingParams


class ModelRunner:
    def __init__(
        self,
        model: str,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int = 4096,
        block_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        dtype: str = "auto",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Lab04 requires CUDA.")

        self.model_path = os.path.expanduser(model)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager

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
        self.hf_config.torch_dtype = torch_dtype
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len")
        self.device = torch.device("cuda", 0)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(self.hf_config)
        load_model(self.model, self.model_path)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def warmup_model(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_model_len = min(self.max_model_len, 1024)
        num_seqs = max(1, min(self.max_num_batched_tokens // max_model_len, self.max_num_seqs))
        warmup_seqs = [
            Sequence([0] * max_model_len, self.block_size, SamplingParams(max_tokens=1))
            for _ in range(num_seqs)
        ]
        self.run(warmup_seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self) -> None:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = self.hf_config.num_key_value_heads
        head_dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        block_bytes = (
            2
            * self.hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()
        )
        self.num_kvcache_blocks = int(total * self.gpu_memory_utilization - used - peak + current) // block_bytes
        if self.num_kvcache_blocks <= 0:
            raise RuntimeError("Unable to reserve KV cache blocks.")
        self.kv_cache = torch.empty(
            2,
            self.hf_config.num_hidden_layers,
            self.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=self.hf_config.torch_dtype,
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
            if not seq.block_table:
                continue
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
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(bucket for bucket in self.graph_bs if bucket >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = min(self.max_num_seqs, 512)
        max_num_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32, device=self.device)
        outputs = torch.zeros(max_bs, self.hf_config.hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }

    def exit(self) -> None:
        if not self.enforce_eager:
            del self.graphs, self.graph_pool, self.graph_vars
        del self.model, self.kv_cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
