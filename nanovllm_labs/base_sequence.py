from __future__ import annotations

from copy import copy
from itertools import count

from nanovllm_labs.sampling_params import SamplingParams


class BaseSequence:
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        sampling_params = SamplingParams() if sampling_params is None else sampling_params
        self.seq_id = next(BaseSequence.counter)
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.arrival_ts = None
        self.scheduled_ts = None
        self.first_token_ts = None
        self.last_token_ts = None
        self.finish_ts = None
        self.itl_sum_s = 0.0
        self.itl_count = 0
        self.finish_reason = None
        self.cached_prompt_tokens = None

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def get_metrics(self) -> dict:
        ttft_s = None if self.arrival_ts is None or self.first_token_ts is None else self.first_token_ts - self.arrival_ts
        queue_s = None if self.arrival_ts is None or self.scheduled_ts is None else self.scheduled_ts - self.arrival_ts
        prefill_s = None if self.scheduled_ts is None or self.first_token_ts is None else self.first_token_ts - self.scheduled_ts
        decode_s = None if self.first_token_ts is None or self.last_token_ts is None else self.last_token_ts - self.first_token_ts
        inference_s = None if self.scheduled_ts is None or self.last_token_ts is None else self.last_token_ts - self.scheduled_ts
        e2e_s = None if self.arrival_ts is None or self.finish_ts is None else self.finish_ts - self.arrival_ts
        if self.first_token_ts is None:
            tpot_s = None
        elif self.num_completion_tokens <= 1:
            tpot_s = 0.0
        else:
            tpot_s = self.itl_sum_s / self.itl_count if self.itl_count > 0 else 0.0
        cached_prompt_tokens = 0 if self.cached_prompt_tokens is None else self.cached_prompt_tokens
        cached_prompt_ratio = 0.0 if self.num_prompt_tokens == 0 else cached_prompt_tokens / self.num_prompt_tokens
        return {
            "queue_ms": None if queue_s is None else queue_s * 1000,
            "compute_ttft_ms": None if prefill_s is None else prefill_s * 1000,
            "ttft_ms": None if ttft_s is None else ttft_s * 1000,
            "decode_ms": None if decode_s is None else decode_s * 1000,
            "inference_ms": None if inference_s is None else inference_s * 1000,
            "tpot_ms": None if tpot_s is None else tpot_s * 1000,
            "e2e_ms": None if e2e_s is None else e2e_s * 1000,
            "prompt_tokens": self.num_prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "cached_prompt_ratio": cached_prompt_ratio,
            "generation_tokens": self.num_completion_tokens,
        }

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
