from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class BenchmarkWorkload:
    num_seqs: int
    prompts: list[list[int]]
    output_lens: list[int]


DEFAULT_NUM_SEQS = 256
DEFAULT_BLOCK_SIZE = 256
MIN_INPUT_LEN = 100
MAX_INPUT_LEN = 1024
MIN_OUTPUT_LEN = 100
MAX_OUTPUT_LEN = 1024


def build_bench_workload(seed: int = 0, num_seqs: int = DEFAULT_NUM_SEQS) -> BenchmarkWorkload:
    rng = Random(seed)
    prompts = [
        [rng.randint(0, 10000) for _ in range(rng.randint(MIN_INPUT_LEN, MAX_INPUT_LEN))]
        for _ in range(num_seqs)
    ]
    output_lens = [rng.randint(MIN_OUTPUT_LEN, MAX_OUTPUT_LEN) for _ in range(num_seqs)]
    return BenchmarkWorkload(num_seqs=num_seqs, prompts=prompts, output_lens=output_lens)


def build_prefix_ratio_workload(
    *,
    seed: int = 0,
    num_seqs: int = DEFAULT_NUM_SEQS,
    prefix_share_ratio_pct: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
    group_size: int = 2,
) -> BenchmarkWorkload:
    if not 0 <= prefix_share_ratio_pct <= 100:
        raise ValueError("prefix_share_ratio_pct must be between 0 and 100")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    base = build_bench_workload(seed=seed, num_seqs=num_seqs)
    if prefix_share_ratio_pct == 0:
        return base

    prompts = [list(prompt) for prompt in base.prompts]
    for group_start in range(0, len(prompts), group_size):
        group_prompts = prompts[group_start:group_start + group_size]
        if not group_prompts:
            continue
        cacheable_blocks = [len(prompt) // block_size for prompt in group_prompts]
        shared_blocks = int(min(cacheable_blocks) * prefix_share_ratio_pct / 100)
        shared_prefix_len = shared_blocks * block_size
        if shared_prefix_len == 0:
            continue

        rng = Random(10_000 + seed + group_start // group_size)
        shared_prefix = [rng.randint(0, 10000) for _ in range(shared_prefix_len)]
        for prompt in group_prompts:
            prompt[:shared_prefix_len] = shared_prefix

    return BenchmarkWorkload(
        num_seqs=base.num_seqs,
        prompts=prompts,
        output_lens=list(base.output_lens),
    )
