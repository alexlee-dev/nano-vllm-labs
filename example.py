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
    parser.add_argument(
        "--model",
        type=str,
        default="~/huggingface/Qwen3-0.6B/",
        help="Local path to the downloaded Qwen3-0.6B model directory. "
             "If omitted, NANOVLLM_MODEL is used.",
    )
    return parser.parse_args()


def default_model_for_lab(lab: int) -> str:
    if lab == 5:
        return "~/huggingface/Qwen3-4B/"
    return "~/huggingface/Qwen3-0.6B/"


def default_tp_for_lab(lab: int) -> int:
    if lab == 5:
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
    model_arg = args.model
    if model_arg == "~/huggingface/Qwen3-0.6B/" and args.lab == 5:
        model_arg = default_model_for_lab(args.lab)
    path = resolve_model_path(model_arg)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    llm_cls = load_llm_engine(args.lab, args.solution)
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else default_tp_for_lab(args.lab)
    llm = llm_cls(path, device="auto", dtype="auto", tensor_parallel_size=tp_size)

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
