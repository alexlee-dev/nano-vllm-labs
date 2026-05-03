from __future__ import annotations

import os
import pickle

import torch
import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from transformers import AutoConfig

from nanovllm_labs.common.loader import load_model
from nanovllm_labs.common.sampler import Sampler
from nanovllm_labs.common.sequence import Sequence
from nanovllm_labs.lab7_solution.models.qwen3 import Qwen3ForCausalLM
from nanovllm_labs.lab7_solution.utils.context import reset_context, set_context
from nanovllm_labs.sampling_params import SamplingParams


def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
    layers_per_partition = num_hidden_layers // pp_size
    partitions = [layers_per_partition for _ in range(pp_size)]
    for i in range(num_hidden_layers % pp_size):
        partitions[i] += 1
    start = sum(partitions[:pp_rank])
    return start, start + partitions[pp_rank]


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
        pipeline_parallel_size: int = 1,
        distributed_init_method: str = "tcp://127.0.0.1:2333",
        shm_name: str = "nanovllm_labs_lab7_pp",
        shm_size_bytes: int = 2**24,
        rank: int = 0,
        event: Event | list[Event] | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Lab07 requires CUDA.")
        if torch.cuda.device_count() < pipeline_parallel_size:
            raise RuntimeError(
                f"Requested pipeline_parallel_size={pipeline_parallel_size}, "
                f"but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )

        self.model_path = os.path.expanduser(model)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = True if pipeline_parallel_size > 1 else enforce_eager
        self.world_size = pipeline_parallel_size
        self.rank = rank
        self.event = event
        self.shm_name = shm_name

        dist.init_process_group(
            "nccl",
            init_method=distributed_init_method,
            world_size=self.world_size,
            rank=self.rank,
        )
        torch.cuda.set_device(self.rank)
        self.device = torch.device("cuda", self.rank)

        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if pipeline_parallel_size > self.hf_config.num_hidden_layers:
            raise ValueError(
                f"pipeline_parallel_size={pipeline_parallel_size} exceeds "
                f"num_hidden_layers={self.hf_config.num_hidden_layers}"
            )
        dtype_map = {
            "auto": torch.bfloat16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype={dtype!r}")
        self.hf_config.torch_dtype = torch_dtype
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len")

        self.start_layer, self.end_layer = get_pp_indices(
            self.hf_config.num_hidden_layers,
            self.rank,
            self.world_size,
        )
        self.num_local_layers = self.end_layer - self.start_layer
        self.is_first_stage = self.rank == 0
        self.is_last_stage = self.rank == self.world_size - 1

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(
            self.hf_config,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            is_first_stage=self.is_first_stage,
            is_last_stage=self.is_last_stage,
        )
        load_model(self.model, self.model_path)
        self.sampler = Sampler()
        self.allocate_kv_cache()
        blocks_t = torch.tensor([self.num_kvcache_blocks], dtype=torch.int64, device=self.device)
        dist.all_reduce(blocks_t, op=dist.ReduceOp.MIN)
        self.num_kvcache_blocks = int(blocks_t.item())
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if self.rank == 0:
                self.shm = SharedMemory(name=self.shm_name, create=True, size=shm_size_bytes)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=self.shm_name)
                self.loop()

    def warmup_model(self) -> None:
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_model_len = min(self.max_model_len, 1024)
        num_seqs = max(1, min(self.max_num_batched_tokens // max_model_len, self.max_num_seqs))
        warmup_seqs = [
            Sequence([0] * max_model_len, self.block_size, SamplingParams(max_tokens=1))
            for _ in range(num_seqs)
        ]
        input_ids, positions = self.prepare_prefill(warmup_seqs)
        hidden_states = None
        residual = None
        if not self.is_first_stage:
            num_tokens = positions.size(0)
            hidden_states = torch.zeros(
                num_tokens,
                self.hf_config.hidden_size,
                dtype=self.hf_config.torch_dtype,
                device=self.device,
            )
            residual = torch.zeros_like(hidden_states)
        hidden_states, _ = self.model(
            positions,
            input_ids=input_ids if self.is_first_stage else None,
            hidden_states=hidden_states,
            residual=residual,
        )
        if self.is_last_stage:
            self.model.compute_logits(hidden_states)
        reset_context()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self) -> None:
        torch.cuda.set_device(self.device)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = self.hf_config.num_key_value_heads
        head_dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        block_bytes = (
            2
            * self.num_local_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()
        )
        available = total * self.gpu_memory_utilization - max(used, current)
        self.num_kvcache_blocks = int(available // block_bytes) if block_bytes else 0
        if self.num_kvcache_blocks <= 0:
            raise RuntimeError("Unable to reserve KV cache blocks.")
        self.kv_cache = torch.empty(
            2,
            self.num_local_layers,
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
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)

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
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
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
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        block_tables_t = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping_t, context_lens=context_lens_t, block_tables=block_tables_t)
        return input_ids_t, positions_t

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)

    def recv_hidden_states(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.empty(
            num_tokens,
            self.hf_config.hidden_size,
            dtype=self.hf_config.torch_dtype,
            device=self.device,
        )
        residual = torch.empty_like(hidden_states)
        dist.recv(hidden_states, src=self.rank - 1)
        dist.recv(residual, src=self.rank - 1)
        return hidden_states, residual

    def send_hidden_states(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> None:
        dist.send(hidden_states.contiguous(), dst=self.rank + 1)
        dist.send(residual.contiguous(), dst=self.rank + 1)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        torch.cuda.set_device(self.device)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        hidden_states = None
        residual = None
        if not self.is_first_stage:
            hidden_states, residual = self.recv_hidden_states(positions.size(0))

        hidden_states, residual = self.model(
            positions,
            input_ids=input_ids if self.is_first_stage else None,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not self.is_last_stage:
            assert residual is not None
            self.send_hidden_states(hidden_states, residual)
            token_ids_t = torch.empty(len(seqs), dtype=torch.int64, device=self.device)
        else:
            temperatures = self.prepare_sample(seqs)
            logits = self.model.compute_logits(hidden_states)
            token_ids_t = self.sampler(logits, temperatures)

        if self.world_size > 1:
            dist.broadcast(token_ids_t, src=self.world_size - 1)
        reset_context()
        return token_ids_t.tolist()

    def read_shm(self) -> tuple[str, list[object]]:
        assert self.world_size > 1 and self.rank > 0
        assert self.event is not None
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name: str, *args: object) -> None:
        assert self.world_size > 1 and self.rank == 0
        assert isinstance(self.event, list)
        data = pickle.dumps([method_name, *args])
        n = len(data)
        capacity = len(self.shm.buf) - 4
        if n > capacity:
            raise RuntimeError(
                f"Shared-memory control payload too large: {n} bytes > {capacity} bytes. "
                "Increase shm_size_bytes or reduce the serialized command payload."
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name: str, *args: object):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name)
        return method(*args)

    def loop(self) -> None:
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def exit(self) -> None:
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        del self.model, self.kv_cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.destroy_process_group()
