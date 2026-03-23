from __future__ import annotations

import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float() / temperatures.unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
