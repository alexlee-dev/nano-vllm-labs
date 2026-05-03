from __future__ import annotations

import os

from transformers import AutoTokenizer

from nanovllm_labs.common.block_manager import BlockManager
from nanovllm_labs.common.scheduler import Scheduler
from nanovllm_labs.common.sequence import Sequence
from nanovllm_labs.lab4_solution.engine.model_runner import ModelRunner
from nanovllm_labs.sampling_params import SamplingParams


class LLMEngine:
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
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.block_manager = BlockManager(self.model_runner.num_kvcache_blocks, block_size)
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            block_manager=self.block_manager,
        )

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
        return self.model_runner.run(seqs, is_prefill=is_prefill)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        return self.scheduler.postprocess(seqs, token_ids)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[dict]:
        del use_tqdm
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

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

    def exit(self) -> None:
        self.model_runner.exit()
