from __future__ import annotations

import atexit
import os
import socket

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from nanovllm_labs.common.block_manager import BlockManager
from nanovllm_labs.common.scheduler import Scheduler
from nanovllm_labs.common.sequence import Sequence
from nanovllm_labs.sampling_params import SamplingParams


class SchedulerLLMEngineBase:
    sequence_cls: type[Sequence] | None = None

    def _init_tokenizer(self, model: str) -> None:
        model = os.path.expanduser(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def _init_scheduler(
        self,
        *,
        num_kvcache_blocks: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> None:
        self.block_manager = BlockManager(num_kvcache_blocks, self.block_size)
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            block_manager=self.block_manager,
        )

    def _normalize_sampling_params(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[SamplingParams]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        return sampling_params

    def _prompt_token_ids(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt)
        return list(prompt)

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> Sequence:
        if self.sequence_cls is None:
            raise TypeError("SchedulerLLMEngineBase subclasses must set sequence_cls.")
        seq = self.sequence_cls(self._prompt_token_ids(prompt), self.block_size, sampling_params)
        self.scheduler.add(seq)
        return seq

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[dict]:
        del use_tqdm
        sampling_params = self._normalize_sampling_params(prompts, sampling_params)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs: dict[int, list[int]] = {}
        while not self.is_finished():
            seqs, is_prefill = self.schedule()
            if not seqs:
                break
            token_ids = self.run_step(seqs, is_prefill=is_prefill)
            for seq_id, out_token_ids in self.postprocess(seqs, token_ids):
                outputs[seq_id] = out_token_ids

        ordered = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        return [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in ordered]


def pick_local_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


class SingleControllerDistributedLLMEngineBase(SchedulerLLMEngineBase):
    sequence_cls = Sequence

    def _init_distributed_engine(
        self,
        *,
        model: str,
        device: str,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        block_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        dtype: str,
        parallel_size: int,
        parallel_size_arg: str,
        runtime_label: str,
        shm_prefix: str,
        runner_cls: type,
        distributed_init_method: str | None = None,
    ) -> None:
        if device not in {"auto", "cuda"}:
            raise ValueError(f"Unsupported device={device!r}")
        if block_size % 256 != 0:
            raise ValueError("block_size must be a multiple of 256")
        if parallel_size < 1:
            raise ValueError(f"{parallel_size_arg} must be >= 1")
        if not torch.cuda.is_available():
            raise RuntimeError(f"{runtime_label} requires CUDA.")
        if torch.cuda.device_count() < parallel_size:
            raise RuntimeError(
                f"Requested {parallel_size_arg}={parallel_size}, "
                f"but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )

        model = os.path.expanduser(model)
        self.block_size = block_size
        self._init_method = distributed_init_method or pick_local_init_method()
        self._shm_name = f"{shm_prefix}_{os.getpid()}"
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
            distributed_init_method=self._init_method,
            shm_name=self._shm_name,
            **{parallel_size_arg: parallel_size},
        )
        for rank in range(1, parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=runner_cls, kwargs={**runner_kwargs, "rank": rank, "event": event})
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = runner_cls(**runner_kwargs, rank=0, event=self.events)
        self._init_tokenizer(model)
        self._init_scheduler(
            num_kvcache_blocks=self.model_runner.num_kvcache_blocks,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        atexit.register(self.exit)

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
