# nano-vllm-labs

Step-by-step labs for learning nanovllm, from a minimal engine to a near-complete runtime.

This repository is an unofficial educational lab series based on nano-vllm.
Original nano-vllm code is Copyright (c) 2025 Xingkai Yu and licensed under the MIT License.
Additional lab materials and modifications are Copyright (c) 2026 Alex Lee and also licensed under the MIT License.

## Lab Overview

- Lab 1 builds the smallest end-to-end LLM inference loop: load a model and tokenizer, generate one token at a time, and decode the result.
- Lab 2 keeps the same simple engine structure, but replaces the Hugging Face model with a hand-written Qwen3 forward pass and custom safetensors weight loading.
- Lab 3 introduces runtime-oriented optimizations: paged KV cache management, request scheduling, and continuous batching across many active sequences.

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

## Run

Use the Make targets to run a lab or its solution version:
```bash
make run-lab1
make run-lab1-s
```

The same naming pattern is available for other labs:
```bash
make run-lab2
make run-lab2-s
make run-lab3
make run-lab3-s
make run-lab4
make run-lab4-s
```

## Benchmark

Use the matching benchmark targets:
```bash
make bench-lab1
make bench-lab1-s
```

The same naming pattern is available for other labs:
```bash
make bench-lab2
make bench-lab2-s
make bench-lab3
make bench-lab3-s
make bench-lab4
make bench-lab4-s
```

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

**Performance Results:**
The `vLLM` and `Nano-vLLM` results below use the 256-sequence benchmark configuration.

| Inference Engine | Command             | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------------|---------------|----------|-----------------------|
| Lab1-solution    | `make bench-lab1-s` | 1,537         | 106.72   | 14.40                 |
| Lab2-solution    | `make bench-lab2-s` | 1,537         | 111.05   | 13.84                 |
| Lab3-solution    | `make bench-lab3-s` | 133,966       | 154.67   | 866.12                |
| vLLM             | N/A                 | 133,966       | 98.37    | 1361.84               |
| Nano-vLLM        | N/A                 | 133,966       | 93.41    | 1434.13               |

