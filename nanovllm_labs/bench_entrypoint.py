from __future__ import annotations

import argparse
from typing import Any, Callable

from nanovllm_labs import SamplingParams
from nanovllm_labs.benchmark_data import BenchmarkWorkload, build_bench_workload
from nanovllm_labs.bench_utils import (
    print_bench_report,
    resolve_model_path,
    run_lab1_bench,
    run_scheduler_bench,
)


ExtraFieldsBuilder = Callable[[argparse.Namespace, str, BenchmarkWorkload], list[tuple[str, Any]]]
LLMBuilder = Callable[[argparse.Namespace, str], Any]
PrefillTokensBuilder = Callable[[list[Any]], int]
BENCH_TEMPERATURE = 0.6


def build_bench_parser(
    *,
    dtype_choices: tuple[str, ...],
    include_device: bool = False,
    include_scheduler_args: bool = False,
    include_max_model_len: bool = False,
    include_enforce_eager: bool = False,
    include_tensor_parallel_size: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    if include_tensor_parallel_size:
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
    if include_device:
        parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=dtype_choices)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=32)
    if include_scheduler_args:
        parser.add_argument("--max-num-seqs", type=int, default=512)
        parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
        if include_max_model_len:
            parser.add_argument("--max-model-len", type=int, default=4096)
        parser.add_argument("--block-size", type=int, default=256)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        if include_enforce_eager:
            parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--bench-timeout-s",
        type=float,
        default=0.0,
        help="Benchmark timeout in seconds. Use 0 to disable the timeout.",
    )
    parser.add_argument("--bench-seed", type=int, default=0)
    return parser


def run_smoke_test(llm: Any, args: argparse.Namespace) -> None:
    smoke = llm.generate(
        [args.prompt],
        SamplingParams(temperature=BENCH_TEMPERATURE, max_tokens=args.max_tokens, ignore_eos=False),
    )[0]
    print(f"smoke_text: {smoke['text']!r}")
    print(f"smoke_tokens: {len(smoke['token_ids'])}")


def build_sampling_params(workload: BenchmarkWorkload) -> list[SamplingParams]:
    return [
        SamplingParams(temperature=BENCH_TEMPERATURE, max_tokens=out_len, ignore_eos=True)
        for out_len in workload.output_lens
    ]


def run_autoregressive_bench_with_workload(
    args: argparse.Namespace,
    *,
    title: str,
    build_llm: LLMBuilder,
    extra_fields: ExtraFieldsBuilder,
    workload: BenchmarkWorkload,
) -> None:
    model = resolve_model_path(args.model)
    llm = build_llm(args, model)

    try:
        run_smoke_test(llm, args)
        stats = run_lab1_bench(
            add_request=llm.add_request,
            step=llm.step,
            eos_token_id=llm.eos_token_id,
            prompts=workload.prompts,
            sampling_params=build_sampling_params(workload),
            timeout_s=args.bench_timeout_s,
        )
        print_bench_report(
            title=title,
            requested_total_tokens=sum(workload.output_lens),
            stats=stats,
            extra_fields=extra_fields(args, model, workload),
            mode="autoregressive",
        )
    finally:
        llm.exit()


def run_autoregressive_entrypoint(
    argv: list[str] | None,
    *,
    title: str,
    bench_num_seqs: int,
    dtype_choices: tuple[str, ...],
    build_llm: LLMBuilder,
    extra_fields: ExtraFieldsBuilder,
    include_device: bool = False,
) -> None:
    parser = build_bench_parser(dtype_choices=dtype_choices, include_device=include_device)
    args = parser.parse_args(argv)
    workload = build_bench_workload(seed=args.bench_seed, num_seqs=bench_num_seqs)
    run_autoregressive_bench_with_workload(
        args,
        title=title,
        build_llm=build_llm,
        extra_fields=extra_fields,
        workload=workload,
    )


def run_scheduler_bench_with_workload(
    args: argparse.Namespace,
    *,
    title: str,
    build_llm: LLMBuilder,
    extra_fields: ExtraFieldsBuilder,
    prefill_tokens_for_step: PrefillTokensBuilder,
    workload: BenchmarkWorkload,
) -> None:
    model = resolve_model_path(args.model)
    llm = build_llm(args, model)

    try:
        run_smoke_test(llm, args)
        stats = run_scheduler_bench(
            add_request=llm.add_request,
            is_finished=llm.is_finished,
            schedule=llm.schedule,
            run_step=llm.run_step,
            postprocess=llm.postprocess,
            prompts=workload.prompts,
            sampling_params=build_sampling_params(workload),
            timeout_s=args.bench_timeout_s,
            prefill_tokens_for_step=prefill_tokens_for_step,
        )
        print_bench_report(
            title=title,
            requested_total_tokens=sum(workload.output_lens),
            stats=stats,
            extra_fields=extra_fields(args, model, workload),
            mode="scheduler",
        )
    finally:
        llm.exit()


def run_scheduler_entrypoint(
    argv: list[str] | None,
    *,
    title: str,
    bench_num_seqs: int,
    dtype_choices: tuple[str, ...],
    build_llm: LLMBuilder,
    extra_fields: ExtraFieldsBuilder,
    prefill_tokens_for_step: PrefillTokensBuilder,
    include_max_model_len: bool = False,
    include_enforce_eager: bool = False,
) -> None:
    parser = build_bench_parser(
        dtype_choices=dtype_choices,
        include_scheduler_args=True,
        include_max_model_len=include_max_model_len,
        include_enforce_eager=include_enforce_eager,
        include_tensor_parallel_size=True,
    )
    args = parser.parse_args(argv)
    workload = build_bench_workload(seed=args.bench_seed, num_seqs=bench_num_seqs)
    run_scheduler_bench_with_workload(
        args,
        title=title,
        build_llm=build_llm,
        extra_fields=extra_fields,
        prefill_tokens_for_step=prefill_tokens_for_step,
        workload=workload,
    )
