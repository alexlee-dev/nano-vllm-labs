import torch
from transformers import AutoConfig

from nanovllm_labs.lab2_solution.engine.sequence import Sequence
from nanovllm_labs.lab2_solution.layers.sampler import Sampler
from nanovllm_labs.lab2_solution.models.qwen3 import Qwen3ForCausalLM
from nanovllm_labs.common.utils.loader import load_model

import os


class ModelRunner:
    def __init__(
        self, 
        model: str,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.model_path = os.path.expanduser(model)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype={dtype!r}")
        self.model_dtype = torch_dtype

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_dtype)
        self.model = Qwen3ForCausalLM(self.hf_config)
        load_model(self.model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.sampler = Sampler()
        torch.set_default_dtype(default_dtype)

    @torch.inference_mode()
    def step(self, seq: Sequence) -> int:
        input_ids = torch.tensor([seq.token_ids], dtype=torch.int64, device=self.device)
        positions = torch.arange(len(seq.token_ids), dtype=torch.int64, device=self.device)
        temperatures = torch.tensor([seq.temperature], dtype=torch.float32, device=self.device)
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)[:, -1, :]
        token_id = self.sampler(logits, temperatures).item()
        seq.append_token(token_id)
        return token_id

    def exit(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
