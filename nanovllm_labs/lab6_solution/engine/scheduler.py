from __future__ import annotations

from collections import deque

from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        block_manager: BlockManager,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule_prefill(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            uncached_prompt_tokens = len(seq) - seq.num_cached_tokens
            if (
                num_batched_tokens + uncached_prompt_tokens > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += uncached_prompt_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
        return scheduled

    def schedule_decode(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        num_seqs = 0
        if not self.running:
            return scheduled
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        if not scheduled:
            return scheduled
        self.running.extendleft(reversed(scheduled))
        return scheduled

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled = self.schedule_prefill()
        if scheduled:
            return scheduled, True
        return self.schedule_decode(), False

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        finished_outputs: list[tuple[int, list[int]]] = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if not seq.ignore_eos and token_id == self.eos_token_id:
                seq.finish_reason = "eos"
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_outputs.append((seq.seq_id, seq.completion_token_ids))
            elif seq.num_completion_tokens == seq.max_tokens:
                seq.finish_reason = "length"
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_outputs.append((seq.seq_id, seq.completion_token_ids))
        return finished_outputs
