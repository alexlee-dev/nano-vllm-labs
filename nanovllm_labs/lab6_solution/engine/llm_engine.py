from __future__ import annotations

import atexit
import os
import queue
import threading
from time import perf_counter
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from nanovllm_labs.lab6_solution.engine.block_manager import BlockManager
from nanovllm_labs.lab6_solution.engine.model_runner import ModelRunner
from nanovllm_labs.lab6_solution.engine.scheduler import Scheduler
from nanovllm_labs.lab6_solution.engine.sequence import Sequence
from nanovllm_labs.sampling_params import SamplingParams


@dataclass
class _StepResult:
    seqs: list[Sequence]
    token_ids: list[int]
    finished: list[tuple[int, list[int]]]
    is_prefill: bool
    prefill_tokens: int
    decode_tokens: int
    start_ts: float
    end_ts: float


def _worker_loop(
    conn,
    rank: int,
    runner_kwargs: dict[str, object],
    scheduler_kwargs: dict[str, object],
) -> None:
    model_runner: ModelRunner | None = None
    try:
        model_runner = ModelRunner(**runner_kwargs, device_id=rank)
        scheduler = Scheduler(
            **scheduler_kwargs,
            block_manager=BlockManager(model_runner.num_kvcache_blocks, runner_kwargs["block_size"]),
        )
        seqs: dict[int, Sequence] = {}
        conn.send(("ready", model_runner.num_kvcache_blocks))

        while True:
            method, payload = conn.recv()
            if method == "add":
                seq = payload
                seq.dp_rank = rank
                seqs[seq.seq_id] = seq
                scheduler.add(seq)
                conn.send(None)
                continue

            if method == "step":
                scheduled = scheduler.schedule_prefill()
                scheduled_is_prefill = bool(scheduled)
                if not scheduled:
                    scheduled = scheduler.schedule_decode()
                    scheduled_is_prefill = False
                snapshots = [
                    (
                        seq.seq_id,
                        seq.num_cached_tokens,
                        seq.num_completion_tokens,
                        scheduled_is_prefill,
                    )
                    for seq in scheduled
                ]
                prefill_tokens = (
                    sum(len(seq) - seq.num_cached_tokens for seq in scheduled)
                    if scheduled_is_prefill
                    else 0
                )
                decode_tokens = 0 if scheduled_is_prefill else len(scheduled)
                start_ts = perf_counter()
                token_ids = (
                    model_runner.run(scheduled, is_prefill=scheduled_is_prefill)
                    if scheduled
                    else []
                )
                finished = scheduler.postprocess(scheduled, token_ids)
                end_ts = perf_counter()
                conn.send(
                    (
                        snapshots,
                        token_ids,
                        [
                            (seq_id, out_token_ids, seqs[seq_id].finish_reason)
                            for seq_id, out_token_ids in finished
                        ],
                        prefill_tokens,
                        decode_tokens,
                        start_ts,
                        end_ts,
                    )
                )
                continue

            if method == "exit":
                conn.send(None)
                break

            raise ValueError(f"Unknown worker method: {method}")
    finally:
        if model_runner is not None:
            model_runner.exit()
        conn.close()


class _RankClient:
    rank: int

    def add_sequence(self, seq: Sequence) -> None:
        raise NotImplementedError

    def step(self) -> _StepResult | None:
        raise NotImplementedError

    def exit(self) -> None:
        raise NotImplementedError


@dataclass
class _RemoteRankClient(_RankClient):
    rank: int
    process: mp.Process
    conn: object
    seqs: dict[int, Sequence]

    def _call(self, method: str, payload: object = None):
        self.conn.send((method, payload))
        return self.conn.recv()

    def add_sequence(self, seq: Sequence) -> None:
        seq.dp_rank = self.rank
        self.seqs[seq.seq_id] = seq
        self._call("add", seq)

    def step(self) -> _StepResult | None:
        snapshots, token_ids, finished, prefill_tokens, decode_tokens, start_ts, end_ts = self._call("step")
        if not snapshots:
            return None
        scheduled: list[Sequence] = []
        is_prefill = False
        for seq_id, num_cached_tokens, prev_num_completion_tokens, seq_is_prefill in snapshots:
            seq = self.seqs[seq_id]
            seq.num_cached_tokens = num_cached_tokens
            seq.scheduled_is_prefill = seq_is_prefill
            seq._prev_num_completion_tokens = prev_num_completion_tokens
            scheduled.append(seq)
            is_prefill = seq_is_prefill
        for seq, token_id in zip(scheduled, token_ids):
            seq.append_token(token_id)
        for seq_id, _out_token_ids, finish_reason in finished:
            self.seqs[seq_id].finish_reason = finish_reason
        return _StepResult(
            seqs=scheduled,
            token_ids=token_ids,
            finished=[(seq_id, out_token_ids) for seq_id, out_token_ids, _ in finished],
            is_prefill=is_prefill,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def exit(self) -> None:
        if not self.process.is_alive():
            self.conn.close()
            return
        self.conn.send(("exit", None))
        self.conn.recv()
        self.process.join()
        self.conn.close()


@dataclass
class _LocalRankClient(_RankClient):
    rank: int
    runner: ModelRunner
    scheduler: Scheduler

    def add_sequence(self, seq: Sequence) -> None:
        seq.dp_rank = self.rank
        self.scheduler.add(seq)

    def step(self) -> _StepResult | None:
        scheduled = self.scheduler.schedule_prefill()
        is_prefill = bool(scheduled)
        if not scheduled:
            scheduled = self.scheduler.schedule_decode()
            is_prefill = False
        if not scheduled:
            return None
        for seq in scheduled:
            seq.scheduled_is_prefill = is_prefill
        prev_counts = {seq.seq_id: seq.num_completion_tokens for seq in scheduled}
        prefill_tokens = (
            sum(len(seq) - seq.num_cached_tokens for seq in scheduled)
            if is_prefill
            else 0
        )
        decode_tokens = 0 if is_prefill else len(scheduled)
        start_ts = perf_counter()
        token_ids = self.runner.run(scheduled, is_prefill=is_prefill)
        finished = self.scheduler.postprocess(scheduled, token_ids)
        end_ts = perf_counter()
        for seq in scheduled:
            seq._prev_num_completion_tokens = prev_counts[seq.seq_id]
        return _StepResult(
            seqs=scheduled,
            token_ids=token_ids,
            finished=finished,
            is_prefill=is_prefill,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def exit(self) -> None:
        self.runner.exit()


class LLMEngine:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_num_seqs: int = 512,
        max_num_batched_tokens: int = 16384,
        max_model_len: int = 4096,
        block_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        dtype: str = "auto",
        data_parallel_size: int = 1,
        **_: object,
    ) -> None:
        if device not in {"auto", "cuda"}:
            raise ValueError(f"Unsupported device={device!r}")
        if data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        if not torch.cuda.is_available():
            raise RuntimeError("Lab06 requires CUDA.")
        num_gpus = torch.cuda.device_count()
        if num_gpus < data_parallel_size:
            raise ValueError(
                f"Requested data_parallel_size={data_parallel_size}, "
                f"but only {num_gpus} CUDA devices are available."
            )

        model = os.path.expanduser(model)
        self.block_size = block_size
        self.data_parallel_size = data_parallel_size
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            model=model,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
        )
        scheduler_kwargs = dict(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
        )

        ctx = mp.get_context("spawn")
        self.ranks: list[_RankClient] = []
        self._result_queue: queue.Queue[_StepResult | None] = queue.Queue()
        self._rank_threads: list[threading.Thread] = []
        self._rank_loops_started = False
        self._active_rank_loops = 0
        self._current_result: _StepResult | None = None
        self._stop_rank_loops = threading.Event()
        self._rank_token_loads = [0] * data_parallel_size
        self._rank_request_counts = [0] * data_parallel_size
        self._seq_costs: dict[int, int] = {}
        self._pending_requests: list[tuple[Sequence, int]] = []

        local_runner = ModelRunner(**runner_kwargs, device_id=0)
        self.ranks.append(
            _LocalRankClient(
                rank=0,
                runner=local_runner,
                scheduler=Scheduler(
                    **scheduler_kwargs,
                    block_manager=BlockManager(local_runner.num_kvcache_blocks, block_size),
                ),
            )
        )

        for rank in range(1, data_parallel_size):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker_loop,
                args=(child_conn, rank, runner_kwargs, scheduler_kwargs),
            )
            process.start()
            method, _num_kvcache_blocks = parent_conn.recv()
            if method != "ready":
                raise RuntimeError(f"DP rank {rank} failed to initialize.")
            self.ranks.append(
                _RemoteRankClient(
                    rank=rank,
                    process=process,
                    conn=parent_conn,
                    seqs={},
                )
            )
        atexit.register(self.exit)

    def _rank_loop(self, rank: _RankClient) -> None:
        try:
            while not self._stop_rank_loops.is_set():
                result = rank.step()
                if result is None:
                    return
                self._result_queue.put(result)
        finally:
            self._result_queue.put(None)

    def _ensure_rank_loops_started(self) -> None:
        if self._rank_loops_started:
            return
        self._flush_pending_requests()
        self._stop_rank_loops.clear()
        self._rank_loops_started = True
        self._active_rank_loops = len(self.ranks)
        for rank in self.ranks:
            thread = threading.Thread(target=self._rank_loop, args=(rank,), daemon=True)
            thread.start()
            self._rank_threads.append(thread)

    def _reset_finished_rank_loops(self) -> None:
        if not self._rank_loops_started or self._active_rank_loops != 0:
            return
        if self._current_result is not None or not self._result_queue.empty():
            return
        for thread in self._rank_threads:
            thread.join(timeout=1)
        self._rank_threads = []
        self._rank_loops_started = False

    def _choose_rank_for_cost(self, cost: int) -> _RankClient:
        rank_id = min(
            range(len(self.ranks)),
            key=lambda idx: (self._rank_token_loads[idx], self._rank_request_counts[idx]),
        )
        self._rank_token_loads[rank_id] += cost
        self._rank_request_counts[rank_id] += 1
        return self.ranks[rank_id]

    def _flush_pending_requests(self) -> None:
        if not self._pending_requests:
            return
        pending = sorted(
            self._pending_requests,
            key=lambda item: item[1],
            reverse=True,
        )
        self._pending_requests = []
        for seq, cost in pending:
            rank = self._choose_rank_for_cost(cost)
            rank.add_sequence(seq)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        self._reset_finished_rank_loops()
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)
        seq = Sequence(prompt_token_ids, self.block_size, sampling_params)
        cost = seq.max_tokens + max(1, seq.num_prompt_tokens // 3)
        self._seq_costs[seq.seq_id] = cost
        if self._rank_loops_started:
            rank = self._choose_rank_for_cost(cost)
            rank.add_sequence(seq)
        else:
            self._pending_requests.append((seq, cost))
        return seq

    def is_finished(self) -> bool:
        if self._pending_requests:
            return False
        if self._rank_loops_started:
            return (
                self._active_rank_loops == 0
                and self._current_result is None
                and self._result_queue.empty()
            )
        return True

    def schedule(self) -> tuple[list[Sequence], bool]:
        self._ensure_rank_loops_started()
        while self._active_rank_loops > 0:
            result = self._result_queue.get()
            if result is None:
                self._active_rank_loops -= 1
                continue
            self._current_result = result
            for seq in result.seqs:
                seq._step_start_ts = result.start_ts
                seq._step_end_ts = result.end_ts
                seq._step_prefill_tokens = result.prefill_tokens
                seq._step_decode_tokens = result.decode_tokens
            return result.seqs, result.is_prefill
        return [], False

    def run_step(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        del seqs, is_prefill
        assert self._current_result is not None
        return self._current_result.token_ids

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        del seqs, token_ids
        assert self._current_result is not None
        finished = self._current_result.finished
        for seq_id, _ in finished:
            cost = self._seq_costs.pop(seq_id, 0)
            rank_id = self._current_result.seqs[0].dp_rank
            self._rank_token_loads[rank_id] = max(0, self._rank_token_loads[rank_id] - cost)
            self._rank_request_counts[rank_id] = max(0, self._rank_request_counts[rank_id] - 1)
        self._current_result = None
        return finished

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[dict]:
        del use_tqdm
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

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

    def exit(self) -> None:
        ranks = getattr(self, "ranks", None)
        if ranks is None:
            return
        self._stop_rank_loops.set()
        for thread in getattr(self, "_rank_threads", []):
            thread.join()
        self._rank_threads = []
        for rank in ranks:
            rank.exit()
        self.ranks = []
