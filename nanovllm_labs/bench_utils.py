from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import floor, ceil
from time import perf_counter
from typing import Any, Callable


@dataclass
class BenchStats:
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    prefill_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_tokens: int = 0
    decode_time_s: float = 0.0
    request_metrics: dict[str, list[float]] = field(default_factory=dict)
    timed_out: bool = False

    def record(self, *, is_prefill: bool, tokens: int, dt: float) -> None:
        if is_prefill:
            self.prefill_tokens += tokens
            self.prefill_time_s += dt
        else:
            self.decode_tokens += tokens
            self.decode_time_s += dt

    @property
    def total_tokens(self) -> int:
        input_tokens = self.prefill_tokens if self.prefill_tokens > 0 else self.prompt_tokens
        return input_tokens + self.decode_tokens


def throughput(tokens: int, dt: float) -> float:
    if dt <= 0:
        return 0.0
    return tokens / dt


def summarize(values: list[float]) -> dict[str, float | int | None]:
    xs = sorted(float(v) for v in values)
    if not xs:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
        }

    def percentile(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q
        lo = floor(pos)
        hi = ceil(pos)
        if lo == hi:
            return xs[lo]
        weight = pos - lo
        return xs[lo] * (1.0 - weight) + xs[hi] * weight

    n = len(xs)
    mid = n // 2
    median = xs[mid] if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])
    return {
        "count": len(xs),
        "mean": sum(xs) / len(xs),
        "median": median,
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def format_summary(name: str, values: list[float]) -> str:
    stats = summarize(values)
    if stats["count"] == 0:
        return f"{name}: count=0"
    return (
        f"{name}: count={stats['count']} "
        f"mean={stats['mean']:.2f} "
        f"median={stats['median']:.2f} "
        f"p95={stats['p95']:.2f} "
        f"p99={stats['p99']:.2f}"
    )


def resolve_model_path(model_arg: str | None) -> str:
    model = model_arg or os.environ.get("NANOVLLM_MODEL")
    if not model:
        raise ValueError(
            "Model path is required. Pass --model /path/to/Qwen3-0.6B "
            "or set NANOVLLM_MODEL."
        )
    model = os.path.expanduser(model)
    if not os.path.isdir(model):
        raise FileNotFoundError(
            f"Model directory not found: {model}. "
            "Download the model locally first."
        )
    return model


def init_request(seq: Any, arrival_ts: float) -> None:
    seq.arrival_ts = arrival_ts


def update_token_timestamps(seq: Any, step_end_ts: float, prev_num_completion_tokens: int, stats: BenchStats) -> None:
    if seq.num_completion_tokens <= prev_num_completion_tokens:
        return
    if seq.first_token_ts is None:
        seq.first_token_ts = step_end_ts
        seq.last_token_ts = step_end_ts
        return
    if seq.last_token_ts is not None:
        seq.itl_sum_s += step_end_ts - seq.last_token_ts
        seq.itl_count += 1
    seq.last_token_ts = step_end_ts


def observe_request_metrics(stats: BenchStats, seq: Any) -> None:
    metrics = seq.get_metrics()
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            stats.request_metrics.setdefault(key, []).append(float(value))


def run_scheduler_bench(
    *,
    add_request: Callable[[list[int], Any], Any],
    is_finished: Callable[[], bool],
    schedule: Callable[[], tuple[list[Any], bool]],
    run_step: Callable[[list[Any], bool], list[int]],
    postprocess: Callable[[list[Any], list[int]], Any],
    prompts: list[list[int]],
    sampling_params: list[Any],
    timeout_s: float,
    prefill_tokens_for_step: Callable[[list[Any]], int],
    decode_tokens_for_step: Callable[[list[Any]], int] | None = None,
) -> BenchStats:
    stats = BenchStats(prompt_tokens=sum(len(prompt_ids) for prompt_ids in prompts))
    for prompt_ids, sp in zip(prompts, sampling_params):
        seq = add_request(prompt_ids, sp)
        init_request(seq, perf_counter())

    t0 = perf_counter()
    deadline = t0 + timeout_s if timeout_s > 0 else None

    while not is_finished():
        if deadline is not None and perf_counter() >= deadline:
            stats.timed_out = True
            break

        seqs, is_prefill = schedule()
        if not seqs:
            break

        prefill_seqs = [
            seq for seq in seqs
            if getattr(seq, "scheduled_is_prefill", is_prefill)
        ]
        decode_seqs = [
            seq for seq in seqs
            if not getattr(seq, "scheduled_is_prefill", is_prefill)
        ]
        async_prefill_tokens = getattr(seqs[0], "_step_prefill_tokens", None)
        async_decode_tokens = getattr(seqs[0], "_step_decode_tokens", None)
        if async_prefill_tokens is not None:
            step_prefill_tokens = async_prefill_tokens
        elif prefill_seqs:
            step_prefill_tokens = prefill_tokens_for_step(prefill_seqs)
        else:
            step_prefill_tokens = 0
        if async_decode_tokens is not None:
            step_decode_tokens = async_decode_tokens
        elif decode_seqs:
            step_decode_tokens = (
                decode_tokens_for_step(decode_seqs)
                if decode_tokens_for_step is not None
                else len(decode_seqs)
            )
        else:
            step_decode_tokens = 0

        async_step_start_ts = getattr(seqs[0], "_step_start_ts", None)
        async_step_end_ts = getattr(seqs[0], "_step_end_ts", None)
        step_t0 = async_step_start_ts if async_step_start_ts is not None else perf_counter()
        for seq in seqs:
            if seq.scheduled_ts is None:
                seq.scheduled_ts = step_t0
            if getattr(seq, "scheduled_is_prefill", is_prefill) and seq.cached_prompt_tokens is None:
                seq.cached_prompt_tokens = getattr(seq, "num_cached_tokens", 0)
        prev_num_completion_tokens = {
            seq.seq_id: getattr(seq, "_prev_num_completion_tokens", seq.num_completion_tokens)
            for seq in seqs
        }
        token_ids = run_step(seqs, is_prefill)
        finished_outputs = postprocess(seqs, token_ids)
        step_end_t = async_step_end_ts if async_step_end_ts is not None else perf_counter()
        step_dt = step_end_t - step_t0
        if step_prefill_tokens:
            stats.record(is_prefill=True, tokens=step_prefill_tokens, dt=step_dt)
        if step_decode_tokens:
            stats.record(is_prefill=False, tokens=step_decode_tokens, dt=step_dt)
        finished_ids = {seq_id for seq_id, _ in finished_outputs}
        for seq in seqs:
            update_token_timestamps(seq, step_end_t, prev_num_completion_tokens[seq.seq_id], stats)
            if seq.seq_id in finished_ids:
                seq.finish_ts = step_end_t
                observe_request_metrics(stats, seq)

        if deadline is not None and perf_counter() >= deadline and not is_finished():
            stats.timed_out = True
            break

    stats.elapsed_s = perf_counter() - t0
    return stats


def run_lab1_bench(
    *,
    add_request: Callable[[list[int], Any], Any],
    step: Callable[[Any], int],
    eos_token_id: int,
    prompts: list[list[int]],
    sampling_params: list[Any],
    timeout_s: float,
) -> BenchStats:
    stats = BenchStats(prompt_tokens=sum(len(prompt_ids) for prompt_ids in prompts))
    t0 = perf_counter()
    deadline = t0 + timeout_s if timeout_s > 0 else None

    for prompt_ids, sp in zip(prompts, sampling_params):
        seq = add_request(prompt_ids, sp)
        init_request(seq, perf_counter())
        seq.scheduled_ts = seq.arrival_ts
        last_step_end = seq.arrival_ts
        finished = False
        for _ in range(seq.max_tokens):
            if deadline is not None and perf_counter() >= deadline:
                stats.timed_out = True
                stats.elapsed_s = perf_counter() - t0
                return stats

            step_t0 = perf_counter()
            prev_num_completion_tokens = seq.num_completion_tokens
            token_id = step(seq)
            step_dt = perf_counter() - step_t0
            step_end_t = step_t0 + step_dt
            stats.record(is_prefill=False, tokens=1, dt=step_dt)
            update_token_timestamps(seq, step_end_t, prev_num_completion_tokens, stats)
            last_step_end = step_end_t
            if (not sp.ignore_eos) and token_id == eos_token_id:
                seq.finish_reason = "eos"
                seq.finish_ts = step_end_t
                observe_request_metrics(stats, seq)
                finished = True
                break
        if not finished:
            seq.finish_reason = "length"
            seq.finish_ts = last_step_end
            observe_request_metrics(stats, seq)

    stats.elapsed_s = perf_counter() - t0
    return stats


def print_bench_report(
    *,
    title: str,
    requested_total_tokens: int,
    stats: BenchStats,
    extra_fields: list[tuple[str, Any]] | None = None,
    mode: str,
) -> None:
    if mode not in {"autoregressive", "scheduler"}:
        raise ValueError(f"Unsupported benchmark report mode: {mode}")
    print(title)
    for key, value in extra_fields or []:
        print(f"{key}: {value}")
    print(f"requested_total_tokens: {requested_total_tokens}")
    if mode == "scheduler":
        print(f"prefill_tokens: {stats.prefill_tokens}")
        print(f"prefill_time_s: {stats.prefill_time_s:.4f}")
        print(f"prefill_throughput_tok_s: {throughput(stats.prefill_tokens, stats.prefill_time_s):.2f}")
    else:
        print(f"prompt_tokens: {stats.prompt_tokens}")
    print(f"decode_tokens: {stats.decode_tokens}")
    print(f"decode_time_s: {stats.decode_time_s:.4f}")
    print(f"decode_throughput_tok_s: {throughput(stats.decode_tokens, stats.decode_time_s):.2f}")
    print(f"total_tokens: {stats.total_tokens}")
    print(f"time_s: {stats.elapsed_s:.4f}")
    print(f"throughput_tok_s: {throughput(stats.decode_tokens, stats.elapsed_s):.2f}")
    print(f"total_throughput_tok_s: {throughput(stats.total_tokens, stats.elapsed_s):.2f}")
    if mode == "scheduler":
        for metric_name in (
            "queue_ms",
            "compute_ttft_ms",
            "ttft_ms",
            "decode_ms",
            "inference_ms",
            "tpot_ms",
            "e2e_ms",
            "prompt_tokens",
            "cached_prompt_tokens",
            "cached_prompt_ratio",
            "generation_tokens",
        ):
            values = stats.request_metrics.get(metric_name, [])
            if values:
                print(format_summary(f"request_{metric_name}", values))
