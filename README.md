# nano-vllm-labs

Step-by-step labs for learning `nano-vllm`, starting from a minimal autoregressive loop and ending at a single-node tensor-parallel serving path.

This repository is an unofficial educational lab series based on `nano-vllm`.
Original `nano-vllm` code is Copyright (c) 2025 Xingkai Yu and licensed under the MIT License.
Additional lab materials and modifications are Copyright (c) 2026 Alex Lee and also licensed under the MIT License.

## What This Repo Covers

The labs are cumulative:

- Lab 1: minimal end-to-end generation loop
- Lab 2: handwritten Qwen3 forward pass plus safetensors loading
- Lab 3: paged KV cache, scheduler, and continuous batching
- Lab 4: warmup-based KV sizing, prefix reuse, and CUDA Graph decode on one GPU
- Lab 5: single-node multi-GPU tensor parallel serving for `Qwen3-4B`

Current scope:

- Labs 1-4 are single-GPU
- Lab 5 is single-node tensor parallel
- Multi-node execution, data parallelism, and pipeline parallelism are still out of scope

## Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For Labs 1-4, set a local `Qwen3-0.6B` path:

```bash
export NANOVLLM_MODEL=~/huggingface/Qwen3-0.6B/
make run-lab4-s
make bench-lab4-s
make bench-prefix-lab4-s
```

For Lab 5, use a local `Qwen3-4B` path:

```bash
python example.py --lab 5 --solution --model ~/huggingface/Qwen3-4B/
make bench-lab5-s
```

`example.py` defaults Lab 5 to `tensor_parallel_size=2` and `Qwen3-4B` if you do not override them.

## Environment Setup

This project expects a local virtual environment at `.venv` because the `Makefile` uses `.venv/bin/python` directly.

### Required Tools

- Python 3.10 to 3.12
- `venv`
- `make`
- `git`
- NVIDIA GPU drivers and a working CUDA environment

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv make git
```

If you want to download models with the Hugging Face CLI:

```bash
python -m pip install "huggingface_hub[cli]"
```

## Model Download

Download `Qwen3-0.6B` for Labs 1-4:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

Download `Qwen3-4B` for Lab 5:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-4B \
  --local-dir ~/huggingface/Qwen3-4B/ \
  --local-dir-use-symlinks False
```

## Run

Student targets:

```bash
make run-lab1
make run-lab2
make run-lab3
make run-lab4
```

Reference solution targets:

```bash
make run-lab1-s
make run-lab2-s
make run-lab3-s
make run-lab4-s
make run-lab5-s
```

Direct entrypoints:

```bash
.venv/bin/python example.py --lab 4 --solution --model ~/huggingface/Qwen3-0.6B/
.venv/bin/python example.py --lab 5 --solution --model ~/huggingface/Qwen3-4B/ --tensor-parallel-size 2
```

## Benchmark

Solution benchmarks:

```bash
make bench-lab1-s
make bench-lab2-s
make bench-lab3-s
make bench-lab4-s
make bench-lab5-s
```

Prefix benchmarks:

```bash
make bench-prefix-lab3-s
make bench-prefix-lab4-s
```

Request-level metrics are meaningful for the scheduler-based labs:

- `request_queue_ms`
- `request_compute_ttft_ms`
- `request_ttft_ms`
- `request_decode_ms`
- `request_inference_ms`
- `request_tpot_ms`
- `request_e2e_ms`

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

## Historical Results

### RTX 4070 Laptop, Qwen3-0.6B

Workload:

- Lab1: 3 sequences
- Lab2: 3 sequences
- Lab3: 256 sequences
- Lab4: 256 sequences
- Input length sampled from `100..1024`
- Output length sampled from `100..1024`

| Inference Engine | Command             | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------------|---------------|----------|-----------------------|
| Lab1-solution    | `make bench-lab1-s` | 1,537         | 106.72   | 14.40                 |
| Lab2-solution    | `make bench-lab2-s` | 1,537         | 111.05   | 13.84                 |
| Lab3-solution    | `make bench-lab3-s` | 133,966       | 154.67   | 866.12                |
| Lab4-solution    | `make bench-lab4-s` | 133,966       | 98.19    | 1364.34               |
| vLLM             | N/A                 | 133,966       | 98.37    | 1361.84               |
| Nano-vLLM        | N/A                 | 133,966       | 93.41    | 1434.13               |

### RTX 5070 Single-GPU, Qwen3-0.6B

Hardware: NVIDIA GeForce RTX 5070, 12GB

| Inference Engine | Command             | Requested Output Tokens | Total Tokens | Time (s) | Output Throughput (tokens/s) |
|------------------|---------------------|--------------------------|--------------|----------|------------------------------|
| Lab3-solution    | `make bench-lab3-s` | 133,966                  | 364,170      | 160.81   | 830.46                       |
| Lab4-solution    | `make bench-lab4-s` | 133,966                  | 333,813      | 45.52    | 2934.00                      |
| Nano-vLLM        | `python bench.py`   | 133,966                  | N/A          | 44.23    | 3029.02                      |

Request-level summary for `Lab4-solution`:

| Metric                    | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|---------------------------|-----------|-------------|----------|----------|
| `request_queue_ms`        | 12821.86  | 13222.43    | 33077.76 | 36526.31 |
| `request_compute_ttft_ms` | 189.85    | 68.28       | 436.71   | 436.71   |
| `request_ttft_ms`         | 13011.71  | 13289.78    | 33144.54 | 36592.90 |
| `request_decode_ms`       | 12741.61  | 12243.62    | 24726.19 | 31166.52 |
| `request_inference_ms`    | 12931.46  | 12477.69    | 25031.06 | 31340.81 |
| `request_tpot_ms`         | 24.89     | 25.14       | 31.96    | 42.99    |
| `request_e2e_ms`          | 25753.32  | 25727.34    | 43180.22 | 44236.63 |

### RTX 5070 Lab4 Data Parallel Result

Hardware: NVIDIA GeForce RTX 5070 x2 (single node, 12GB each)

Workload: default 256-sequence benchmark with random input/output lengths in `100..1024`

**Throughput Summary:**

| Model | Inference Engine | Command | Requested Output Tokens | Total Tokens | Time (s) | Output Throughput (tokens/s) | Total Throughput (tokens/s) |
|-------|------------------|---------|--------------------------|--------------|----------|------------------------------|-----------------------------|
| `Qwen3-0.6B` | Lab4-solution (DP=2) | `python bench.py --lab 4 --solution --data-parallel-size 2` | 133,966 | 356,816 | 24.78 | 5389.65 | 14397.46 |
| `Qwen3-4B` | Lab4-solution (DP=2) | `python bench.py --lab 4 --solution --model ~/huggingface/Qwen3-4B --data-parallel-size 2` | 133,966 | 299,549 | 645.61 | 206.82 | 463.98 |

`lab4_solution` implements replicated data parallelism for dense models. For backwards compatibility,
`--tensor-parallel-size 2` is accepted by the lab4 solution as DP=2, but new runs should use
`--data-parallel-size 2`.

**Qwen3-4B DP=2 Request-Level Metrics:**

| Metric                         | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|--------------------------------|-----------|-------------|----------|----------|
| `request_queue_ms`             | 312303.70 | 321213.20   | 593169.79 | 621317.77 |
| `request_compute_ttft_ms`      | 172.91    | 137.80      | 267.62   | 1150.58 |
| `request_ttft_ms`              | 312476.61 | 321369.64   | 593409.14 | 621483.10 |
| `request_decode_ms`            | 12350.95  | 12307.16    | 25191.51 | 29714.04 |
| `request_inference_ms`         | 12523.86  | 12442.95    | 25421.03 | 29908.49 |
| `request_tpot_ms`              | 24.73     | 20.26       | 44.80    | 65.27 |
| `request_e2e_ms`               | 324827.56 | 335803.30   | 607565.55 | 629283.97 |

### RTX 5070 Qwen3-4B Multi-GPU Result

Hardware: NVIDIA GeForce RTX 5070 x2, 12GB each

This is a single-machine tensor-parallel comparison, not a multi-node run.

| Inference Engine     | Command                                                            | Requested Output Tokens | Total Tokens | Time (s) | Output Throughput (tokens/s) |
|----------------------|--------------------------------------------------------------------|--------------------------|--------------|----------|------------------------------|
| Lab5-solution (TP=1) | `python bench.py --lab 5 --solution --tensor-parallel-size 1`      | 133,966                  | 324,092      | 304.15   | 439.00                       |
| Lab5-solution (TP=2) | `make bench-lab5-s`                                                | 133,966                  | 329,208      | 112.25   | 1189.83                      |

Mean request-level comparison:

| Metric                    | TP=1 Mean (ms) | TP=2 Mean (ms) |
|---------------------------|----------------|----------------|
| `request_queue_ms`        | 141350.31      | 33689.63       |
| `request_compute_ttft_ms` | 199.43         | 961.81         |
| `request_ttft_ms`         | 141549.74      | 34651.44       |
| `request_decode_ms`       | 10701.95       | 29974.35       |
| `request_inference_ms`    | 10901.37       | 30936.16       |
| `request_tpot_ms`         | 20.80          | 59.85          |
| `request_e2e_ms`          | 152251.69      | 64625.78       |

Why TP helps for Lab 5:

- `Qwen3-4B` is large enough that single-GPU memory pressure dominates queueing behavior.
- TP=2 makes each steady-state decode token more expensive, but it reduces queue pressure enough to win end to end.
- The main benefit is restored concurrency and model fit, not a cheaper per-token decode path.

## Prefix Benchmark

The prefix benchmark isolates prompt-prefix reuse:

- `0%` shared prefix: no reuse baseline
- `100%` shared prefix: maximal reuse case

The most important outputs are:

- `request_cached_prompt_tokens`
- `request_cached_prompt_ratio`

## Lab Docs

- [Lab 1](nanovllm_labs/lab1/README.md)
- [Lab 2](nanovllm_labs/lab2/README.md)
- [Lab 3](nanovllm_labs/lab3/README.md)
- [Lab 4](nanovllm_labs/lab4/engine/README.md)
- [Lab 5](nanovllm_labs/lab5/README.md)

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=alexlee-dev/nano-vllm-labs&type=date&logscale&legend=top-left)](https://www.star-history.com/?repos=alexlee-dev%2Fnano-vllm-labs&type=date&logscale=&legend=top-left)
