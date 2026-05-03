from __future__ import annotations

import os
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Callable

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm_labs.common.sequence import Sequence
from nanovllm_labs.sampling_params import SamplingParams


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype_map = {
        "auto": torch.bfloat16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype={dtype!r}")
    return torch_dtype


class ModelRunnerBase:
    def _init_model_runner_base(
        self,
        *,
        model: str,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        block_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        dtype: str,
        device: torch.device,
        set_context: Callable[..., None],
        reset_context: Callable[[], None],
    ) -> None:
        self.model_path = os.path.expanduser(model)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.device = device
        self._set_context = set_context
        self._reset_context = reset_context

        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.hf_config.torch_dtype = resolve_torch_dtype(dtype)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len")

    def _to_device_tensor(self, data: object, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(self.device, non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return self._to_device_tensor(block_tables, dtype=torch.int32)

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
            positions.extend(range(seq.num_cached_tokens, seqlen))
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
                slot_mapping.extend(range(start, end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids_t = self._to_device_tensor(input_ids, dtype=torch.int64)
        positions_t = self._to_device_tensor(positions, dtype=torch.int64)
        cu_seqlens_q_t = self._to_device_tensor(cu_seqlens_q, dtype=torch.int32)
        cu_seqlens_k_t = self._to_device_tensor(cu_seqlens_k, dtype=torch.int32)
        slot_mapping_t = self._to_device_tensor(slot_mapping, dtype=torch.int32)
        self._set_context(
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
        input_ids_t = self._to_device_tensor(input_ids, dtype=torch.int64)
        positions_t = self._to_device_tensor(positions, dtype=torch.int64)
        slot_mapping_t = self._to_device_tensor(slot_mapping, dtype=torch.int32)
        context_lens_t = self._to_device_tensor(context_lens, dtype=torch.int32)
        block_tables_t = self.prepare_block_tables(seqs)
        self._set_context(
            False,
            slot_mapping=slot_mapping_t,
            context_lens=context_lens_t,
            block_tables=block_tables_t,
        )
        return input_ids_t, positions_t

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return self._to_device_tensor(temperatures, dtype=torch.float32)

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
        self.run(warmup_seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def reset_context(self) -> None:
        self._reset_context()


class SharedMemoryModelRunnerMixin:
    world_size: int
    rank: int
    event: Event | list[Event] | None
    shm: SharedMemory

    def _init_shared_command_channel(self, *, shm_name: str, shm_size_bytes: int) -> None:
        self.shm_name = shm_name
        if self.world_size <= 1:
            return
        if self.rank == 0:
            self.shm = SharedMemory(name=self.shm_name, create=True, size=shm_size_bytes)
            dist.barrier()
            return
        dist.barrier()
        self.shm = SharedMemory(name=self.shm_name)
        self.loop()

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
