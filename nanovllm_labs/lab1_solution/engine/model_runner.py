from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM

from .sequence import Sequence


class ModelRunner:
    """Owns the model and performs one decoding step at a time.

    In Lab 1 this runner is intentionally simple:
    - one request at a time
    - one full forward pass per generated token
    - no KV cache
    - no batching

    This is slower than a production runtime, but it makes the control flow
    easy to inspect and reason about.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        model = os.path.expanduser(model)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype == "auto":
            # Qwen3 checkpoints use bf16 by default on GPU; CPU stays in fp32.
            torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype={dtype!r}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        # Temperature rescales the logits before sampling.
        # Higher temperature -> flatter distribution -> more randomness.
        scaled = logits.float() / max(temperature, 1e-5)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    @torch.inference_mode()
    def step(self, seq: Sequence) -> int:
        # The model expects a batch dimension, so one sequence becomes shape
        # [1, seq_len]. Lab 1 re-feeds the entire history every step.
        input_ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        # use_cache=False is intentional here: Lab 1 demonstrates the simplest
        # possible decoding loop before introducing KV cache in later labs.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # logits[0, -1, :] means:
        # - batch item 0
        # - last time step
        # - probability scores over the whole vocabulary
        token_id = self._sample(outputs.logits[0, -1, :], seq.temperature)

        # Mutate the sequence in place so the next decoding step sees the new
        # token as part of the context.
        seq.append_token(token_id)
        return token_id

    def exit(self) -> None:
        # Explicit cleanup is not strictly necessary in small scripts, but it
        # makes the lifecycle of the engine clearer, especially on GPU.
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
