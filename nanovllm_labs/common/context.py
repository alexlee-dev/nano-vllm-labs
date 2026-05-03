from __future__ import annotations

from dataclasses import dataclass
from threading import local

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


class ContextStore:
    def __init__(self, *, thread_local: bool = False) -> None:
        self._thread_local = thread_local
        self._state = local() if thread_local else None
        self._context = Context()

    def get_context(self) -> Context:
        if not self._thread_local:
            return self._context
        context = getattr(self._state, "context", None)
        if context is None:
            context = Context()
            self._state.context = context
        return context

    def set_context(
        self,
        is_prefill: bool,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        slot_mapping: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
    ) -> None:
        context = Context(
            is_prefill,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
        )
        if not self._thread_local:
            self._context = context
        else:
            self._state.context = context

    def reset_context(self) -> None:
        if not self._thread_local:
            self._context = Context()
        else:
            self._state.context = Context()
