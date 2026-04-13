# nano-vllm-labs

Step-by-step labs for learning nanovllm, from a minimal engine to a near-complete runtime.

This repository is an unofficial educational lab series based on nano-vllm.
Original nano-vllm code is Copyright (c) 2025 Xingkai Yu and licensed under the MIT License.
Additional lab materials and modifications are Copyright (c) 2026 Alex Lee and also licensed under the MIT License.

## Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
export NANOVLLM_MODEL=~/huggingface/Qwen3-0.6B/
make run-lab4-s
make bench-lab4-s
make bench-prefix-lab4-s
```

## Scope

This lab series currently stops at a single-GPU serving engine.

That is a deliberate scope limit for this repository: I only have one GPU available for development and validation, so the code here focuses on the parts that can be built and benchmarked cleanly on one card.

If you want the fuller production-oriented runtime, including code beyond this single-GPU lab scope, refer directly to the original nano-vllm project.

## Lab Overview

- Lab 1 builds the smallest end-to-end LLM inference loop: load a model and tokenizer, generate one token at a time, and decode the result.
- Lab 2 keeps the same simple engine structure, but replaces the Hugging Face model with a hand-written Qwen3 forward pass and custom safetensors weight loading.
- Lab 3 introduces runtime-oriented optimizations: paged KV cache management, request scheduling, and continuous batching across many active sequences.
- Lab 4 keeps the Lab 3 paged-KV runtime, then pushes the single-GPU execution path further with cleaner runtime state tracking, warmup-based memory sizing, and CUDA Graph capture for decode batches.

## Environment Setup

This project expects a local virtual environment at `.venv` because the `Makefile` uses `.venv/bin/python` directly.

### Required Tools

- Python 3.10 to 3.12
- `venv`
- `make`
- `git`
- NVIDIA GPU drivers and a working CUDA environment for GPU execution

On Ubuntu/Debian, you can install the basic system tools with:
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv make git
```

### Create and Populate `.venv`

Create the virtual environment in the repository root:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If you want to download models with the CLI shown below, install the Hugging Face CLI in the same environment:
```bash
python -m pip install "huggingface_hub[cli]"
```

After `.venv` is created and dependencies are installed, the `make` targets in this repo should work directly.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

Then point the scripts at that local model directory:
```bash
export NANOVLLM_MODEL=~/huggingface/Qwen3-0.6B/
```

## Run

The student `lab1` to `lab4` packages are starter skeletons for the exercises.
They are not expected to run successfully until you fill in the missing code.

If you want a working reference implementation, use the solution targets:
```bash
make run-lab1-s
make run-lab2-s
make run-lab3-s
make run-lab4-s
```

If you want to work through the exercises, use the matching student targets:
```bash
make run-lab1
make run-lab2
make run-lab3
make run-lab4
```

The same model path can also be passed explicitly without `make`:
```bash
.venv/bin/python example.py --lab 4 --solution --model ~/huggingface/Qwen3-0.6B/
```

## Benchmark

The published numbers below come from the solution implementations.
Use the matching solution benchmark targets if you want the same code path:
```bash
make bench-lab1-s
make bench-lab2-s
make bench-lab3-s
make bench-lab4-s
```

The student benchmark targets exist for parity with the lab structure, but they only make sense after you implement the corresponding lab.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests:
  - Lab1: 3 sequences
  - Lab2: 3 sequences
  - Lab3: 256 sequences
  - Lab4: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

For `lab3` and `lab4`, the benchmark also prints request-level latency summaries in the form:
`request_<metric>: count=... mean=... median=... p95=... p99=...`

These are production-style serving metrics rather than only aggregate throughput.

Definitions:
- `request_queue_ms`: arrival to first scheduling
- `request_compute_ttft_ms`: first scheduling to first output token
- `request_ttft_ms`: arrival to first output token
- `request_decode_ms`: first output token to last output token
- `request_inference_ms`: first scheduling to last output token
- `request_tpot_ms`: average per-token latency after the first token
- `request_e2e_ms`: arrival to final request completion

Relationships:
- `request_ttft_ms = request_queue_ms + request_compute_ttft_ms`
- `request_inference_ms = request_compute_ttft_ms + request_decode_ms`

These request-level metrics are only meaningful for the scheduler-based labs (`lab3` and `lab4`).
`lab1` and `lab2` still print the aggregate throughput-oriented metrics, but do not expose the same queueing/runtime breakdown.

**Performance Results:**
The `vLLM` and `Nano-vLLM` results below use the 256-sequence benchmark configuration.
Measured on: RTX 4070 Laptop GPU (8GB), `Qwen3-0.6B`, default benchmark workload with random input/output lengths in `100..1024`.

| Inference Engine | Command             | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------------|---------------|----------|-----------------------|
| Lab1-solution    | `make bench-lab1-s` | 1,537         | 106.72   | 14.40                 |
| Lab2-solution    | `make bench-lab2-s` | 1,537         | 111.05   | 13.84                 |
| Lab3-solution    | `make bench-lab3-s` | 133,966       | 154.67   | 866.12                |
| Lab4-solution    | `make bench-lab4-s` | 133,966       | 98.19    | 1364.34               |
| vLLM             | N/A                 | 133,966       | 98.37    | 1361.84               |
| Nano-vLLM        | N/A                 | 133,966       | 93.41    | 1434.13               |

### Prefix Benchmark

Use the prefix benchmark to isolate the effect of prompt-prefix reuse:

```bash
make bench-prefix-lab3-s
make bench-prefix-lab4-s
```

This benchmark only applies to `lab3` and `lab4`.
It runs `0%` and `100%` shared-prefix workloads so the comparison stays clean: `0%` is the no-reuse baseline, and `100%` is the prefix-reuse case.

The most important prefix-specific outputs are:
- `request_cached_prompt_tokens`
- `request_cached_prompt_ratio`

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=alexlee-dev/nano-vllm-labs&type=date&logscale&legend=top-left)](https://www.star-history.com/?repos=alexlee-dev%2Fnano-vllm-labs&type=date&logscale=&legend=top-left)
