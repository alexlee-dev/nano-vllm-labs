from __future__ import annotations

import os

from transformers import AutoTokenizer


class SchedulerLLMEngineBase:
    """Common scheduler-driven engine helpers reused by Labs 4-6."""

    sequence_cls = None

    def _init_tokenizer(self, model: str) -> None:
        model = os.path.expanduser(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def _normalize_sampling_params(self, prompts, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        return sampling_params

    def _prompt_token_ids(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt)
        return list(prompt)

    def add_request(self, prompt: str | list[int], sampling_params):
        if self.sequence_cls is None:
            raise TypeError("SchedulerLLMEngineBase subclasses must set sequence_cls.")
        seq = self.sequence_cls(self._prompt_token_ids(prompt), self.block_size, sampling_params)
        self.scheduler.add(seq)
        return seq

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params,
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
