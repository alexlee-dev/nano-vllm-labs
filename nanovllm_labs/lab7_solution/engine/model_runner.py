from __future__ import annotations

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event

from nanovllm_labs.common.loader import load_model
from nanovllm_labs.common.model_runner import ModelRunnerBase, SharedMemoryModelRunnerMixin
from nanovllm_labs.common.sampler import Sampler
from nanovllm_labs.lab7_solution.models.qwen3 import Qwen3ForCausalLM
from nanovllm_labs.lab7_solution.utils.context import reset_context, set_context


def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
    layers_per_partition = num_hidden_layers // pp_size
    partitions = [layers_per_partition for _ in range(pp_size)]
    for i in range(num_hidden_layers % pp_size):
        partitions[i] += 1
    start = sum(partitions[:pp_rank])
    return start, start + partitions[pp_rank]


class ModelRunner(SharedMemoryModelRunnerMixin, ModelRunnerBase):
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

        self.world_size = pipeline_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl",
            init_method=distributed_init_method,
            world_size=self.world_size,
            rank=self.rank,
        )
        torch.cuda.set_device(self.rank)
        self._init_model_runner_base(
            model=model,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True if pipeline_parallel_size > 1 else enforce_eager,
            dtype=dtype,
            device=torch.device("cuda", self.rank),
            set_context=set_context,
            reset_context=reset_context,
        )
        if pipeline_parallel_size > self.hf_config.num_hidden_layers:
            raise ValueError(
                f"pipeline_parallel_size={pipeline_parallel_size} exceeds "
                f"num_hidden_layers={self.hf_config.num_hidden_layers}"
            )

        self.start_layer, self.end_layer = get_pp_indices(
            self.hf_config.num_hidden_layers,
            self.rank,
            self.world_size,
        )
        self.num_local_layers = self.end_layer - self.start_layer
        self.is_first_stage = self.rank == 0
        self.is_last_stage = self.rank == self.world_size - 1

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.hf_config.torch_dtype)
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
        self._init_shared_command_channel(shm_name=shm_name, shm_size_bytes=shm_size_bytes)

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
    def run(self, seqs, is_prefill: bool) -> list[int]:
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
        self.reset_context()
        return token_ids_t.tolist()

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
