from __future__ import annotations

import argparse

from nanovllm_labs.bench_entrypoint import build_bench_parser, run_scheduler_bench_with_workload
from nanovllm_labs.benchmark_data import build_prefix_ratio_workload
from nanovllm_labs.bench_specs import BenchSpec, get_bench_spec


SUPPORTED_LABS = {3, 4}


def parse_ratios(raw: str) -> list[int]:
    if raw == "all":
        return [0, 100]
    ratio = int(raw)
    if ratio not in {0, 100}:
        raise ValueError("prefix_share_ratio_pct must be one of: 0, 100, all")
    return [ratio]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, BenchSpec]:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--lab", type=int, required=True, choices=sorted(SUPPORTED_LABS))
    base_parser.add_argument("--solution", action="store_true")
    base_args, _ = base_parser.parse_known_args(argv)

    spec = get_bench_spec(lab=base_args.lab, solution=base_args.solution)
    if spec.kind != "scheduler":
        raise ValueError("Prefix benchmark only supports scheduler labs.")

    parser = build_bench_parser(
        dtype_choices=spec.dtype_choices,
        include_scheduler_args=True,
        include_max_model_len=spec.include_max_model_len,
        include_enforce_eager=spec.include_enforce_eager,
    )
    parser.add_argument("--lab", type=int, required=True, choices=sorted(SUPPORTED_LABS))
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--prefix-share-ratio-pct", default="all")
    parser.add_argument("--prefix-group-size", type=int, default=2)
    return parser.parse_args(argv), spec


def run_ratio(args: argparse.Namespace, spec: BenchSpec, ratio: int) -> None:
    workload = build_prefix_ratio_workload(
        seed=args.bench_seed,
        num_seqs=spec.bench_num_seqs,
        prefix_share_ratio_pct=ratio,
        block_size=args.block_size,
        group_size=args.prefix_group_size,
    )
    run_scheduler_bench_with_workload(
        args,
        title=f"{spec.title.removesuffix('_bench')}_prefix_{ratio}pct_bench",
        build_llm=spec.build_llm,
        extra_fields=lambda ns, model, wl: spec.extra_fields(ns, model, wl) + [
            ("prefix_share_ratio_pct", ratio),
            ("prefix_group_size", ns.prefix_group_size),
        ],
        prefill_tokens_for_step=spec.prefill_tokens_for_step,
        workload=workload,
    )


def main(argv: list[str] | None = None) -> None:
    args, spec = parse_args(argv)
    for idx, ratio in enumerate(parse_ratios(args.prefix_share_ratio_pct)):
        if idx:
            print()
        run_ratio(args, spec, ratio)


if __name__ == "__main__":
    main()
