from __future__ import annotations

import atexit
import os
import socket

from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm_labs.common.engine.llm_engine import SchedulerLLMEngineBase
from nanovllm_labs.common.runtime.block_manager import BlockManager
from nanovllm_labs.lab5_solution.engine.model_runner import ModelRunner
from nanovllm_labs.common.runtime.scheduler import Scheduler
from nanovllm_labs.common.runtime.sequence import Sequence
from nanovllm_labs.sampling_params import SamplingParams


def _pick_local_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


class LLMEngine(SchedulerLLMEngineBase):
    sequence_cls = Sequence

    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_num_seqs: int = 512,
        max_num_batched_tokens: int = 16384,
        max_model_len: int = 4096,
        block_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        distributed_init_method: str | None = None,
        **_: object,
    ) -> None:
        if device not in {"auto", "cuda"}:
            raise ValueError(f"Unsupported device={device!r}")
        if block_size % 256 != 0:
            raise ValueError("block_size must be a multiple of 256")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not torch.cuda.is_available():
            raise RuntimeError("Lab04 requires CUDA.")
        if torch.cuda.device_count() < tensor_parallel_size:
            raise RuntimeError(
                f"Requested tensor_parallel_size={tensor_parallel_size}, "
                f"but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )

        model = os.path.expanduser(model)
        self.block_size = block_size
        self.tensor_parallel_size = tensor_parallel_size
        self._init_method = distributed_init_method or _pick_local_init_method()
        self._shm_name = f"nanovllm_labs_lab4_tp_{os.getpid()}"
        self.ps: list[mp.Process] = []
        self.events: list[object] = []

        ctx = mp.get_context("spawn")
        runner_kwargs = dict(
            model=model,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            distributed_init_method=self._init_method,
            shm_name=self._shm_name,
        )
        for rank in range(1, tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, kwargs={**runner_kwargs, "rank": rank, "event": event})
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(**runner_kwargs, rank=0, event=self.events)
        self._init_tokenizer(model)
        self.block_manager = BlockManager(self.model_runner.num_kvcache_blocks, block_size)
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            block_manager=self.block_manager,
        )
        atexit.register(self.exit)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)
        seq = Sequence(prompt_token_ids, self.block_size, sampling_params)
        self.scheduler.add(seq)
        return seq

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def schedule(self) -> tuple[list[Sequence], bool]:
        return self.scheduler.schedule()

    def run_step(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return self.model_runner.call("run", seqs, is_prefill)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        return self.scheduler.postprocess(seqs, token_ids)

    def exit(self) -> None:
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return
        model_runner.call("exit")
        del self.model_runner
        for process in self.ps:
            process.join()
        self.ps.clear()
