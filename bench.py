from __future__ import annotations

import argparse
import importlib
import os
import time
from random import randint, seed

from nanovllm_labs import SamplingParams


NUM_SEQS_BY_LAB = {
    1: 3,
    2: 3,
    3: 256,
    4: 256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--solution", action="store_true")
    return parser.parse_args()


def num_seqs_for_lab(lab: int) -> int:
    return NUM_SEQS_BY_LAB.get(lab, 256)


def load_llm_engine(lab: int, solution: bool):
    module_name = f"nanovllm_labs.lab{lab}{'_solution' if solution else ''}.engine.llm_engine"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Engine module not found for lab={lab}, solution={solution}: {module_name}"
        ) from exc
    return module.LLMEngine


def build_llm(lab: int, solution: bool, model: str):
    llm_cls = load_llm_engine(lab, solution)
    match lab:
        case 1:
            return llm_cls(model)
        case 2:
            return llm_cls(model, max_model_len=4096, enforce_eager=False)
        case 3:
            return llm_cls(model, max_model_len=4096, enforce_eager=False)
        case 4:
            return llm_cls(model, max_model_len=4096, enforce_eager=False)
    raise ValueError(f"unsupported lab: {lab}")


def get_num_output_tokens(output) -> int:
    if isinstance(output, dict) and "token_ids" in output:
        return len(output["token_ids"])
    raise TypeError(f"unsupported benchmark output type: {type(output)!r}")


def main() -> None:
    args = parse_args()
    seed(0)
    num_seqs = num_seqs_for_lab(args.lab)
    max_input_len = 1024
    max_output_len = 1024

    model = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = build_llm(args.lab, args.solution, model)
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(get_num_output_tokens(output) for output in outputs)
    throughput = total_tokens / t
    suffix = "-solution" if args.solution else ""
    print(f"Lab{args.lab}{suffix}: Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
