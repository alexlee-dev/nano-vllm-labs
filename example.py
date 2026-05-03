import argparse
import importlib
import os

from transformers import AutoTokenizer

from nanovllm_labs import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=int, default=1)
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="~/huggingface/Qwen3-0.6B/",
        help="Local path to the downloaded Qwen3-0.6B model directory. "
             "If omitted, NANOVLLM_MODEL is used.",
    )
    return parser.parse_args()


def default_model_for_lab(lab: int) -> str:
    if lab in {5, 7}:
        return "~/huggingface/Qwen3-4B/"
    return "~/huggingface/Qwen3-0.6B/"


def default_tp_for_lab(lab: int) -> int:
    if lab == 5:
        return 2
    return 1


def default_dp_for_lab(lab: int) -> int:
    del lab
    return 1


def default_pp_for_lab(lab: int) -> int:
    if lab == 7:
        return 2
    return 1


def load_llm_engine(lab: int, solution: bool):
    module_name = f"nanovllm_labs.lab{lab}{'_solution' if solution else ''}.engine.llm_engine"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Engine module not found for lab={lab}, solution={solution}: {module_name}"
        ) from exc
    return module.LLMEngine


def get_completion_text(output):
    if isinstance(output, dict):
        return output["text"]
    return str(output)


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


def main():
    args = parse_args()
    if args.lab == 6 and args.tensor_parallel_size is not None:
        raise ValueError("Lab 6 does not support tensor_parallel_size; use --data-parallel-size.")
    if args.lab == 7 and args.tensor_parallel_size is not None:
        raise ValueError("Lab 7 does not support tensor_parallel_size; use --pipeline-parallel-size.")
    if args.lab == 7 and args.data_parallel_size is not None:
        raise ValueError("Lab 7 does not support data_parallel_size; use --pipeline-parallel-size.")
    model_arg = args.model
    if model_arg == "~/huggingface/Qwen3-0.6B/" and args.lab in {5, 7}:
        model_arg = default_model_for_lab(args.lab)
    path = resolve_model_path(model_arg)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    llm_cls = load_llm_engine(args.lab, args.solution)
    llm_kwargs = {
        "device": "auto",
        "dtype": "auto",
    }
    if args.lab == 5:
        llm_kwargs["tensor_parallel_size"] = (
            args.tensor_parallel_size
            if args.tensor_parallel_size is not None
            else default_tp_for_lab(args.lab)
        )
    if args.lab == 6:
        llm_kwargs["data_parallel_size"] = (
            args.data_parallel_size
            if args.data_parallel_size is not None
            else default_dp_for_lab(args.lab)
        )
    if args.lab == 7:
        llm_kwargs["pipeline_parallel_size"] = (
            args.pipeline_parallel_size
            if args.pipeline_parallel_size is not None
            else default_pp_for_lab(args.lab)
        )
    llm = llm_cls(
        path,
        **llm_kwargs,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {get_completion_text(output)!r}")

    llm.exit()


if __name__ == "__main__":
    main()
