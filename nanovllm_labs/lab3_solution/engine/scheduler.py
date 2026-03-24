from collections import deque

from nanovllm_labs.lab3_solution.engine.block_manager import BlockManager
from nanovllm_labs.lab3_solution.engine.sequence import Sequence


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

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled: list[Sequence] = []
        num_batched_tokens = 0
        while self.waiting and len(self.running) + len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            uncached_prompt_tokens = len(seq) - seq.num_cached_tokens
            if (
                num_batched_tokens + uncached_prompt_tokens > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            self.waiting.popleft()
            self.block_manager.allocate(seq)
            self.running.append(seq)
            scheduled.append(seq)
            num_batched_tokens += uncached_prompt_tokens
        if scheduled:
            return scheduled, True
        
        decode_batch: list[Sequence] = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                decode_batch.append(seq)
                num_seqs += 1
        self.running.extendleft(reversed(decode_batch))
        return decode_batch, False
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        finished_outputs: list[tuple[int, list[int]]] = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos_token_id) or seq.num_completion_tokens == seq.max_tokens:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_outputs.append((seq.id, seq.completion_token_ids))
        return finished_outputs
