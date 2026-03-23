import argparse
import importlib
import os

from transformers import AutoTokenizer

from nanovllm_labs import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=int, default=1)
    parser.add_argument("--solution", action="store_true")
    return parser.parse_args()


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


def main():
    args = parse_args()
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    llm_cls = load_llm_engine(args.lab, args.solution)
    llm = llm_cls(path, device="auto", dtype="auto")

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
