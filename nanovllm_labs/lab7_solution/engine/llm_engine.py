from __future__ import annotations

from nanovllm_labs.common.engine.llm_engine import SingleControllerDistributedLLMEngineBase
from nanovllm_labs.lab7_solution.engine.model_runner import ModelRunner


class LLMEngine(SingleControllerDistributedLLMEngineBase):
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
        pipeline_parallel_size: int = 1,
        distributed_init_method: str | None = None,
        **_: object,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self._init_distributed_engine(
            model=model,
            device=device,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            parallel_size=pipeline_parallel_size,
            parallel_size_arg="pipeline_parallel_size",
            runtime_label="Lab07",
            shm_prefix="nanovllm_labs_lab7_pp",
            runner_cls=ModelRunner,
            distributed_init_method=distributed_init_method,
        )
