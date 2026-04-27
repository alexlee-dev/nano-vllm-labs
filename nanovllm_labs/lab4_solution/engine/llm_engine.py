from __future__ import annotations

import atexit
import os
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from nanovllm_labs.lab4_solution.engine.block_manager import BlockManager
from nanovllm_labs.lab4_solution.engine.model_runner import ModelRunner
from nanovllm_labs.lab4_solution.engine.scheduler import Scheduler
from nanovllm_labs.lab4_solution.engine.sequence import Sequence
from nanovllm_labs.sampling_params import SamplingParams


def _snapshot_sequence(seq: Sequence) -> tuple[int, int, list[int]]:
    return seq.seq_id, seq.num_cached_tokens, list(seq.block_table)


def _worker_loop(conn, runner_kwargs: dict[str, object]) -> None:
    model_runner = ModelRunner(**runner_kwargs)
    seqs: dict[int, Sequence] = {}
    try:
        while True:
            method, payload = conn.recv()
            if method == "add":
                seq = payload
                seqs[seq.seq_id] = seq
                conn.send(None)
                continue
            if method == "run":
                snapshots, is_prefill = payload
                local_seqs: list[Sequence] = []
                for seq_id, num_cached_tokens, block_table in snapshots:
                    seq = seqs[seq_id]
                    seq.num_cached_tokens = num_cached_tokens
                    seq.block_table = block_table
                    local_seqs.append(seq)
                token_ids = model_runner.run(local_seqs, is_prefill=is_prefill)
                for seq, token_id in zip(local_seqs, token_ids):
                    seq.append_token(token_id)
                conn.send(token_ids)
                continue
            if method == "exit":
                conn.send(None)
                break
            raise ValueError(f"Unknown worker method: {method}")
    finally:
        model_runner.exit()
        conn.close()


class _RunnerClient:
    def add_sequence(self, seq: Sequence) -> None:
        raise NotImplementedError

    def run_step(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        raise NotImplementedError

    def exit(self) -> None:
        raise NotImplementedError


class _LocalRunnerClient(_RunnerClient):
    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner

    def add_sequence(self, seq: Sequence) -> None:
        del seq

    def run_step(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return self.runner.run(seqs, is_prefill=is_prefill)

    def exit(self) -> None:
        self.runner.exit()


class _RemoteRunnerClient(_RunnerClient):
    def __init__(self, process: mp.Process, conn) -> None:
        self.process = process
        self.conn = conn

    def add_sequence(self, seq: Sequence) -> None:
        self.conn.send(("add", seq))
        self.conn.recv()

    def send_run(self, seqs: list[Sequence], is_prefill: bool) -> None:
        snapshots = [_snapshot_sequence(seq) for seq in seqs]
        self.conn.send(("run", (snapshots, is_prefill)))

    def recv_run(self) -> list[int]:
        return self.conn.recv()

    def run_step(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self.send_run(seqs, is_prefill)
        return self.recv_run()

    def exit(self) -> None:
        if not self.process.is_alive():
            return
        self.conn.send(("exit", None))
        self.conn.recv()
        self.process.join()
        self.conn.close()


@dataclass
class _DeviceWorker:
    device_id: int
    runner: _RunnerClient
    scheduler: Scheduler

    def pending_load(self) -> int:
        return len(self.scheduler.waiting) + len(self.scheduler.running)

    def add(self, seq: Sequence) -> None:
        seq.dp_worker_id = self.device_id
        self.scheduler.add(seq)
        self.runner.add_sequence(seq)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def schedule_prefill(self) -> list[Sequence]:
        return self.scheduler.schedule_prefill()

    def schedule_decode(self) -> list[Sequence]:
        return self.scheduler.schedule_decode()

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        return self.scheduler.postprocess(seqs, token_ids)

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
        tensor_parallel_size: int = 1,
        **_: object,
    ) -> None:
        if device not in {"auto", "cuda"}:
            raise ValueError(f"Unsupported device={device!r}")
        model = os.path.expanduser(model)
        self.block_size = block_size
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        num_gpus = torch.cuda.device_count()
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if num_gpus < tensor_parallel_size:
            raise ValueError(
                f"Requested {tensor_parallel_size} workers, but only {num_gpus} CUDA devices are available."
            )
        self.num_workers = tensor_parallel_size
        self._processes: list[mp.Process] = []

        ctx = mp.get_context("spawn")
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
        self.workers: list[_DeviceWorker] = []

        local_runner = ModelRunner(**runner_kwargs, device_id=0)
        local_block_manager = BlockManager(local_runner.num_kvcache_blocks, block_size)
        self.workers.append(
            _DeviceWorker(
                device_id=0,
                runner=_LocalRunnerClient(local_runner),
                scheduler=Scheduler(
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    eos_token_id=self.eos_token_id,
                    block_manager=local_block_manager,
                ),
            )
        )

        for device_id in range(1, self.num_workers):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker_loop,
                args=(child_conn, {**runner_kwargs, "device_id": device_id}),
            )
            process.start()
            self._processes.append(process)
            self.workers.append(
                _DeviceWorker(
                    device_id=device_id,
                    runner=_RemoteRunnerClient(process, parent_conn),
                    scheduler=Scheduler(
                        max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_num_batched_tokens,
                        eos_token_id=self.eos_token_id,
                        block_manager=BlockManager(local_runner.num_kvcache_blocks, block_size),
                    ),
                )
            )
        atexit.register(self.exit)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)
        seq = Sequence(prompt_token_ids, self.block_size, sampling_params)
        worker = min(self.workers, key=lambda item: item.pending_load())
        worker.add(seq)
        return seq

    def is_finished(self) -> bool:
        return all(worker.is_finished() for worker in self.workers)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled: list[Sequence] = []
        for worker in self.workers:
            scheduled.extend(worker.schedule_prefill())
        if scheduled:
            return scheduled, True

        for worker in self.workers:
            scheduled.extend(worker.schedule_decode())
        return scheduled, False

    def run_step(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        seqs_by_worker: dict[int, list[Sequence]] = {}
        for seq in seqs:
            seqs_by_worker.setdefault(seq.dp_worker_id, []).append(seq)

        for worker_id, worker_seqs in seqs_by_worker.items():
            runner = self.workers[worker_id].runner
            if isinstance(runner, _RemoteRunnerClient):
                runner.send_run(worker_seqs, is_prefill)

        token_ids_by_seq: dict[int, int] = {}
        for worker_id, worker_seqs in seqs_by_worker.items():
            runner = self.workers[worker_id].runner
            if isinstance(runner, _RemoteRunnerClient):
                worker_token_ids = runner.recv_run()
            else:
                worker_token_ids = runner.run_step(worker_seqs, is_prefill)
            for seq, token_id in zip(worker_seqs, worker_token_ids):
                token_ids_by_seq[seq.seq_id] = token_id
        return [token_ids_by_seq[seq.seq_id] for seq in seqs]

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int]]]:
        seqs_by_worker: dict[int, list[Sequence]] = {}
        token_ids_by_worker: dict[int, list[int]] = {}
        for seq, token_id in zip(seqs, token_ids):
            seqs_by_worker.setdefault(seq.dp_worker_id, []).append(seq)
            token_ids_by_worker.setdefault(seq.dp_worker_id, []).append(token_id)

        finished_outputs: list[tuple[int, list[int]]] = []
        for worker_id, worker_seqs in seqs_by_worker.items():
            finished_outputs.extend(self.workers[worker_id].postprocess(worker_seqs, token_ids_by_worker[worker_id]))
        return finished_outputs

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
        workers = getattr(self, "workers", None)
        if workers is None:
            return
        for worker in workers:
            worker.exit()
        self.workers = []
