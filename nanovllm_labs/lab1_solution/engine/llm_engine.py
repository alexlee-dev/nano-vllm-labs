from __future__ import annotations

import os

from transformers import AutoTokenizer

from nanovllm_labs.sampling_params import SamplingParams

from .model_runner import ModelRunner
from .sequence import Sequence


class LLMEngine:
    """High-level orchestration layer for Lab 1 generation.

    LLMEngine does not execute tensor operations itself. Instead, it manages
    the generation workflow:
    - tokenize inputs
    - create request state objects
    - call the model runner step by step
    - stop when the request finishes
    - decode the generated tokens back to text
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        model = os.path.expanduser(model)
        self.model_runner = ModelRunner(model=model, device=device, dtype=dtype)

        # The tokenizer converts user-facing text into token ids and back.
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            # Many causal models do not define a separate pad token. Reusing
            # EOS is a practical fallback for simple experiments.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        # Support both raw text prompts and pre-tokenized prompts.
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)
        return Sequence(prompt_token_ids, sampling_params)

    def step(self, seq: Sequence) -> int:
        # Delegate one decoding step to the lower-level runner.
        return self.model_runner.step(seq)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            # One SamplingParams object can be broadcast to all prompts.
            sampling_params = [sampling_params] * len(prompts)

        outputs: list[dict] = []
        for prompt, sp in zip(prompts, sampling_params):
            seq = self.add_request(prompt, sp)

            # Decode one token at a time until we hit the configured budget or
            # the model emits EOS.
            for _ in range(sp.max_tokens):
                token_id = self.step(seq)
                if (not sp.ignore_eos) and token_id == self.eos_token_id:
                    break

            # completion_token_ids excludes the original prompt tokens, so the
            # returned text contains only newly generated content.
            outputs.append(
                {
                    "text": self.tokenizer.decode(seq.completion_token_ids),
                    "token_ids": seq.completion_token_ids,
                }
            )
        return outputs

    def exit(self) -> None:
        # Forward cleanup to the runner, which owns the actual model object.
        self.model_runner.exit()
