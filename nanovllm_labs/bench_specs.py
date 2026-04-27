from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Literal

from nanovllm_labs.bench_entrypoint import run_autoregressive_entrypoint, run_scheduler_entrypoint


ExtraFieldsBuilder = Callable[[argparse.Namespace, str, Any], list[tuple[str, Any]]]
LLMBuilder = Callable[[argparse.Namespace, str], Any]
PrefillTokensBuilder = Callable[[list[Any]], int]


@dataclass(frozen=True)
class BenchSpec:
    lab: int
    solution: bool
    kind: Literal["autoregressive", "scheduler"]
    title: str
    bench_num_seqs: int
    dtype_choices: tuple[str, ...]
    build_llm: LLMBuilder
    extra_fields: ExtraFieldsBuilder
    include_device: bool = False
    include_max_model_len: bool = False
    include_enforce_eager: bool = False
    prefill_tokens_for_step: PrefillTokensBuilder | None = None

def load_llm_engine(module_name: str):
    return importlib.import_module(module_name).LLMEngine


def build_title(*, lab: int, solution: bool) -> str:
    return f"lab{lab}{'_solution' if solution else ''}_bench"


def _autoregressive_fields(include_device: bool) -> ExtraFieldsBuilder:
    del include_device

    def build_fields(args: argparse.Namespace, model: str, workload: Any) -> list[tuple[str, Any]]:
        del args, model, workload
        return []

    return build_fields


def _lab1_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "dtype": args.dtype,
    }


def _lab2_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dtype": args.dtype,
    }


def _lab3_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dtype": args.dtype,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }


def _lab4_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
    }


def make_autoregressive_spec(
    *,
    lab: int,
    solution: bool,
    module_name: str,
    dtype_choices: tuple[str, ...],
    kwargs_builder: Callable[[argparse.Namespace], dict[str, Any]],
    include_device: bool = False,
) -> BenchSpec:
    return BenchSpec(
        lab=lab,
        solution=solution,
        kind="autoregressive",
        title=build_title(lab=lab, solution=solution),
        bench_num_seqs=3,
        dtype_choices=dtype_choices,
        include_device=include_device,
        build_llm=lambda args, model: load_llm_engine(module_name)(model=model, **kwargs_builder(args)),
        extra_fields=_autoregressive_fields(include_device),
    )


def make_scheduler_spec(
    *,
    lab: int,
    solution: bool,
    module_name: str,
    dtype_choices: tuple[str, ...],
    kwargs_builder: Callable[[argparse.Namespace], dict[str, Any]],
    extra_fields: ExtraFieldsBuilder,
    prefill_tokens_for_step: PrefillTokensBuilder,
    include_max_model_len: bool = False,
    include_enforce_eager: bool = False,
) -> BenchSpec:
    return BenchSpec(
        lab=lab,
        solution=solution,
        kind="scheduler",
        title=build_title(lab=lab, solution=solution),
        bench_num_seqs=256,
        dtype_choices=dtype_choices,
        build_llm=lambda args, model: load_llm_engine(module_name)(model=model, **kwargs_builder(args)),
        extra_fields=extra_fields,
        include_max_model_len=include_max_model_len,
        include_enforce_eager=include_enforce_eager,
        prefill_tokens_for_step=prefill_tokens_for_step,
    )


BENCH_SPECS: dict[tuple[int, bool], BenchSpec] = {
    (1, False): make_autoregressive_spec(
        lab=1,
        solution=False,
        module_name="nanovllm_labs.lab1.engine.llm_engine",
        dtype_choices=("auto", "float16", "float32"),
        include_device=True,
        kwargs_builder=_lab1_kwargs,
    ),
    (1, True): make_autoregressive_spec(
        lab=1,
        solution=True,
        module_name="nanovllm_labs.lab1_solution.engine.llm_engine",
        dtype_choices=("auto", "float16", "float32"),
        include_device=True,
        kwargs_builder=_lab1_kwargs,
    ),
    (2, False): make_autoregressive_spec(
        lab=2,
        solution=False,
        module_name="nanovllm_labs.lab2.engine.llm_engine",
        dtype_choices=("auto", "float16", "bfloat16", "float32"),
        kwargs_builder=_lab2_kwargs,
    ),
    (2, True): make_autoregressive_spec(
        lab=2,
        solution=True,
        module_name="nanovllm_labs.lab2_solution.engine.llm_engine",
        dtype_choices=("auto", "float16", "bfloat16", "float32"),
        kwargs_builder=_lab2_kwargs,
    ),
    (3, False): make_scheduler_spec(
        lab=3,
        solution=False,
        module_name="nanovllm_labs.lab3.engine.llm_engine",
        dtype_choices=("auto", "float16", "float32"),
        kwargs_builder=_lab3_kwargs,
        extra_fields=lambda args, model, workload: [],
        prefill_tokens_for_step=lambda seqs: sum(len(seq.prompt_token_ids) - seq.num_cached_tokens for seq in seqs),
    ),
    (3, True): make_scheduler_spec(
        lab=3,
        solution=True,
        module_name="nanovllm_labs.lab3_solution.engine.llm_engine",
        dtype_choices=("auto", "float16", "float32"),
        kwargs_builder=_lab3_kwargs,
        extra_fields=lambda args, model, workload: [],
        prefill_tokens_for_step=lambda seqs: sum(len(seq.prompt_token_ids) - seq.num_cached_tokens for seq in seqs),
    ),
    (4, False): make_scheduler_spec(
        lab=4,
        solution=False,
        module_name="nanovllm_labs.lab4.engine.llm_engine",
        dtype_choices=("auto", "float16", "bfloat16", "float32"),
        include_max_model_len=True,
        include_enforce_eager=True,
        kwargs_builder=_lab4_kwargs,
        extra_fields=lambda args, model, workload: [],
        prefill_tokens_for_step=lambda seqs: sum(seq.num_prompt_tokens - seq.num_cached_tokens for seq in seqs),
    ),
    (4, True): make_scheduler_spec(
        lab=4,
        solution=True,
        module_name="nanovllm_labs.lab4_solution.engine.llm_engine",
        dtype_choices=("auto", "float16", "bfloat16", "float32"),
        include_max_model_len=True,
        include_enforce_eager=True,
        kwargs_builder=_lab4_kwargs,
        extra_fields=lambda args, model, workload: [],
        prefill_tokens_for_step=lambda seqs: sum(seq.num_prompt_tokens - seq.num_cached_tokens for seq in seqs),
    ),
    (5, True): make_scheduler_spec(
        lab=5,
        solution=True,
        module_name="nanovllm_labs.lab5_solution.engine.llm_engine",
        dtype_choices=("auto", "float16", "bfloat16", "float32"),
        include_max_model_len=True,
        include_enforce_eager=True,
        kwargs_builder=_lab4_kwargs,
        extra_fields=lambda args, model, workload: [],
        prefill_tokens_for_step=lambda seqs: sum(seq.num_prompt_tokens - seq.num_cached_tokens for seq in seqs),
    ),
}


def get_bench_spec(*, lab: int, solution: bool) -> BenchSpec:
    try:
        return BENCH_SPECS[(lab, solution)]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark spec for lab={lab}, solution={solution}") from exc


def run_bench_spec(argv: list[str] | None, *, lab: int, solution: bool) -> None:
    spec = get_bench_spec(lab=lab, solution=solution)
    argv = list(argv or [])
    if lab == 5 and "--model" not in argv:
        argv = ["--model", "~/huggingface/Qwen3-4B/"] + argv
    if lab == 5 and "--tensor-parallel-size" not in argv:
        argv = ["--tensor-parallel-size", "2"] + argv
    if spec.kind == "autoregressive":
        run_autoregressive_entrypoint(
            argv,
            title=spec.title,
            bench_num_seqs=spec.bench_num_seqs,
            dtype_choices=spec.dtype_choices,
            include_device=spec.include_device,
            build_llm=spec.build_llm,
            extra_fields=spec.extra_fields,
        )
        return

    run_scheduler_entrypoint(
        argv,
        title=spec.title,
        bench_num_seqs=spec.bench_num_seqs,
        dtype_choices=spec.dtype_choices,
        build_llm=spec.build_llm,
        extra_fields=spec.extra_fields,
        prefill_tokens_for_step=spec.prefill_tokens_for_step,
        include_max_model_len=spec.include_max_model_len,
        include_enforce_eager=spec.include_enforce_eager,
    )
