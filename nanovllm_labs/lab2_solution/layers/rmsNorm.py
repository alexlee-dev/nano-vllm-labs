from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(ms + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
            residual = x
        out = self._rms(x).to(self.weight.dtype) * self.weight
        if residual is None:
            return out.to(x.dtype)
        return out.to(x.dtype), residual
