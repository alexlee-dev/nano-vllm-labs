import os

from transformers import AutoTokenizer

from nanovllm_labs.lab2_solution.engine.model_runner import ModelRunner
from nanovllm_labs.lab2_solution.engine.sequence import Sequence
from nanovllm_labs.sampling_params import SamplingParams


class LLMEngine:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
        **_: object,
    ) -> None:
        model = os.path.expanduser(model)
        self.model_runner = ModelRunner(model=model, device=device, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)
        return Sequence(prompt_token_ids, sampling_params)

    def step(self, seq: Sequence) -> int:
        return self.model_runner.step(seq)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        outputs: list[dict] = []
        for prompt, sampling_param in zip(prompts, sampling_params):
            seq = self.add_request(prompt, sampling_param)
            for _ in range(sampling_param.max_tokens):
                token_id = self.step(seq)
                if (not sampling_param.ignore_eos) and token_id == self.eos_token_id:
                    break
            outputs.append(
                {
                    "text": self.tokenizer.decode(seq.completion_token_ids),
                    "token_ids": seq.completion_token_ids,
                }
            )
        return outputs
    
    def exit(self) -> None:
        self.model_runner.exit()
