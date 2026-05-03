from __future__ import annotations

import os

from nanovllm_labs.common.engine.llm_engine import SchedulerLLMEngineBase
from nanovllm_labs.common.sequence import Sequence
from nanovllm_labs.lab4_solution.engine.model_runner import ModelRunner


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
        **_: object,
    ) -> None:
        if device not in {"auto", "cuda"}:
            raise ValueError(f"Unsupported device={device!r}")
        model = os.path.expanduser(model)
        self.block_size = block_size
        self.model_runner = ModelRunner(
            model=model,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
        )
        self._init_tokenizer(model)
        self._init_scheduler(
            num_kvcache_blocks=self.model_runner.num_kvcache_blocks,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def schedule(self) -> tuple[list[Sequence], bool]:
        return self.scheduler.schedule()

    def run_step(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return self.model_runner.run(seqs, is_prefill=is_prefill)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        return self.scheduler.postprocess(seqs, token_ids)

    def exit(self) -> None:
        self.model_runner.exit()
