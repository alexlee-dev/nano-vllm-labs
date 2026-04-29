# nano-vllm-labs

Step-by-step labs for learning `nano-vllm`, starting from a minimal autoregressive loop and ending at small single-node multi-GPU serving paths.

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
- Lab 6: single-node replicated data parallel serving across full dense model replicas

Current scope:

- Labs 1-4 are single-GPU
- Lab 5 is single-node tensor parallel
- Lab 6 is single-node replicated data parallel
- Multi-node execution and pipeline parallelism are still out of scope

The project is intentionally educational rather than production-complete. If you want the fuller production-oriented runtime, refer directly to the original `nano-vllm` project.

## Lab Overview

- Lab 1 builds the smallest end-to-end LLM inference loop: load a model and tokenizer, generate one token at a time, and decode the result.
- Lab 2 keeps the same simple engine structure, but replaces the Hugging Face model with a hand-written Qwen3 forward pass and custom safetensors weight loading.
- Lab 3 introduces runtime-oriented optimizations: paged KV cache management, request scheduling, and continuous batching across many active sequences.
- Lab 4 keeps the Lab 3 paged-KV runtime, then pushes the single-GPU execution path further with cleaner runtime state tracking, warmup-based memory sizing, and CUDA Graph capture for decode batches.
- Lab 5 adds tensor parallelism for sharding a larger dense model across GPUs.
- Lab 6 adds replicated data parallelism so multiple GPUs can serve independent requests concurrently.

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

For Lab 6, the default benchmark target uses `Qwen3-0.6B` with replicated data parallel workers:

```bash
export NANOVLLM_MODEL=~/huggingface/Qwen3-0.6B/
python example.py --lab 6 --solution --data-parallel-size 2
make run-lab6-s
make bench-lab6-s
```

If you want to benchmark the larger dense-model case explicitly:

```bash
python example.py --lab 6 --solution --model ~/huggingface/Qwen3-4B/ --data-parallel-size 2
python bench.py --lab 6 --solution --model ~/huggingface/Qwen3-4B/ --data-parallel-size 2
```

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

Download `Qwen3-4B` for Labs 5-6:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-4B \
  --local-dir ~/huggingface/Qwen3-4B/ \
  --local-dir-use-symlinks False
```

## Run

If you want a working reference implementation, use the solution targets:

```bash
make run-lab1-s
make run-lab2-s
make run-lab3-s
make run-lab4-s
make run-lab5-s
make run-lab6-s
```

If you want to work through the exercises, use the matching student targets:

```bash
make run-lab1
make run-lab2
make run-lab3
make run-lab4
```

Direct entrypoints:

```bash
.venv/bin/python example.py --lab 4 --solution --model ~/huggingface/Qwen3-0.6B/
.venv/bin/python example.py --lab 5 --solution --model ~/huggingface/Qwen3-4B/ --tensor-parallel-size 2
.venv/bin/python example.py --lab 6 --solution --data-parallel-size 2
```

## Benchmark

Solution benchmarks:

```bash
make bench-lab1-s
make bench-lab2-s
make bench-lab3-s
make bench-lab4-s
make bench-lab5-s
make bench-lab6-s
```

Prefix benchmarks:

```bash
make bench-prefix-lab3-s
make bench-prefix-lab4-s
```

Request-level metrics are meaningful for the scheduler-based labs.

For `lab3` through `lab6`, the benchmark also prints request-level latency summaries in the form:
`request_<metric>: count=... mean=... median=... p95=... p99=...`

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

These are example measurements from local runs on the listed hardware. They are useful for shape and comparison, but exact numbers will vary by driver stack, PyTorch version, and background load.

These request-level metrics are only meaningful for the scheduler-based labs (`lab3` through `lab6`).
`lab1` and `lab2` still print the aggregate throughput-oriented metrics, but do not expose the same queueing/runtime breakdown.

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

These runs all use the same `Qwen3-0.6B` model on one GPU:

| Inference Engine | Command             | Requested Output Tokens | Total Tokens | Time (s) | Output Throughput (tokens/s) | Total Throughput (tokens/s) |
|------------------|---------------------|--------------------------|--------------|----------|------------------------------|-----------------------------|
| Lab1-solution    | `make bench-lab1-s` | 1,537                    | 3,709        | 79.94    | 19.23                        | 46.40                       |
| Lab2-solution    | `make bench-lab2-s` | 1,537                    | 3,709        | 89.26    | 17.22                        | 41.55                       |
| Lab3-solution    | `make bench-lab3-s` | 133,966                  | 364,170      | 173.98   | 767.63                       | 2093.21                     |
| Lab4-solution    | `make bench-lab4-s` | 133,966                  | 333,813      | 45.47    | 2937.25                      | 7341.38                     |
| Nano-vLLM        | `python bench.py`   | 133,966                  | N/A          | 44.23    | 3029.02                      | N/A                         |

Do not compare the default Lab 5 numbers below directly against this table: Lab 5 switches to `Qwen3-4B` and defaults to `tensor_parallel_size=2`.

Request-level summary for `Lab4-solution`:

| Metric                    | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|---------------------------|-----------|-------------|----------|----------|
| `request_queue_ms`        | 12798.04  | 13191.14    | 33019.78 | 36469.41 |
| `request_compute_ttft_ms` | 190.11    | 67.50       | 440.61   | 440.61   |
| `request_ttft_ms`         | 12988.15  | 13258.44    | 33086.06 | 36535.86 |
| `request_decode_ms`       | 12722.75  | 12226.40    | 24665.65 | 31108.64 |
| `request_inference_ms`    | 12912.86  | 12456.02    | 24969.86 | 31282.52 |
| `request_tpot_ms`         | 24.86     | 25.09       | 31.90    | 42.88    |
| `request_e2e_ms`          | 25710.90  | 25669.88    | 43125.85 | 44182.49 |

### RTX 5070 x2, Single-Node TP, Qwen3-4B

Hardware: NVIDIA GeForce RTX 5070 x2, 12GB each

This is a single-machine tensor-parallel comparison, not a multi-node run.

These runs use `Qwen3-4B`, so they are best compared against other `Qwen3-4B` runs, not against the `Qwen3-0.6B` Lab 4 table above.

| Inference Engine     | Command                                                       | Requested Output Tokens | Total Tokens | Time (s) | Output Throughput (tokens/s) |
|----------------------|---------------------------------------------------------------|--------------------------|--------------|----------|------------------------------|
| Lab5-solution (TP=1) | `python bench.py --lab 5 --solution --tensor-parallel-size 1` | 133,966                  | 324,092      | 304.15   | 439.00                       |
| Lab5-solution (TP=2) | `make bench-lab5-s`                                           | 133,966                  | 329,208      | 110.76   | 1205.76                      |

Mean request-level comparison:

| Metric                    | TP=1 Mean (ms) | TP=2 Mean (ms) |
|---------------------------|----------------|----------------|
| `request_queue_ms`        | 141350.31      | 33733.96       |
| `request_compute_ttft_ms` | 199.43         | 964.46         |
| `request_ttft_ms`         | 141549.74      | 34698.42       |
| `request_decode_ms`       | 10701.95       | 29585.91       |
| `request_inference_ms`    | 10901.37       | 30550.37       |
| `request_tpot_ms`         | 20.80          | 59.21          |
| `request_e2e_ms`          | 152251.69      | 64284.33       |

Why TP helps for Lab 5:

- `Qwen3-4B` is large enough that single-GPU memory pressure dominates queueing behavior.
- TP=2 makes each steady-state decode token more expensive, but it reduces queue pressure enough to win end to end.
- The main benefit is restored concurrency and model fit, not a cheaper per-token decode path.

### RTX 5070 x2, Single-Node DP, Qwen3-0.6B and Qwen3-4B

Hardware: NVIDIA GeForce RTX 5070 x2, 12GB each

This is replicated dense data parallelism: each GPU holds a full model copy and serves independent requests.

Use this table when you want a same-algorithm comparison against single-GPU Lab 4 on a model that already fits on one card. If the model does not fit on one card, you still need Lab 5 tensor parallelism instead.

| Model       | Inference Engine     | Command                                                                    | Requested Output Tokens | Total Tokens | Time (s) | Output Throughput (tokens/s) | Total Throughput (tokens/s) |
|-------------|----------------------|----------------------------------------------------------------------------|--------------------------|--------------|----------|------------------------------|-----------------------------|
| `Qwen3-0.6B` | Lab6-solution (DP=2) | `python bench.py --lab 6 --solution --data-parallel-size 2`                | 133,966                  | 356,816      | 25.64    | 5208.87                      | 13914.54                    |
| `Qwen3-4B`   | Lab6-solution (DP=2) | `python bench.py --lab 6 --solution --model ~/huggingface/Qwen3-4B --data-parallel-size 2` | 133,966 | 299,549 | 645.61 | 206.82 | 463.98 |

`lab6_solution` implements replicated data parallelism for dense models, so Lab 6 runs should use `--data-parallel-size`.

For small models that fit comfortably on one GPU, DP often wins on throughput because the forward path does not pay TP communication costs. For larger models that do not fit on one GPU, TP remains the way to make the model runnable at all.

Request-level summary for `Qwen3-4B` on `Lab6-solution (DP=2)`:

| Metric                    | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|---------------------------|-----------|-------------|----------|----------|
| `request_queue_ms`        | 312303.70 | 321213.20   | 593169.79 | 621317.77 |
| `request_compute_ttft_ms` | 172.91    | 137.80      | 267.62   | 1150.58 |
| `request_ttft_ms`         | 312476.61 | 321369.64   | 593409.14 | 621483.10 |
| `request_decode_ms`       | 12350.95  | 12307.16    | 25191.51 | 29714.04 |
| `request_inference_ms`    | 12523.86  | 12442.95    | 25421.03 | 29908.49 |
| `request_tpot_ms`         | 24.73     | 20.26       | 44.80    | 65.27 |
| `request_e2e_ms`          | 324827.56 | 335803.30   | 607565.55 | 629283.97 |

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
- [Lab 6](nanovllm_labs/lab6_solution/README.md)

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=alexlee-dev/nano-vllm-labs&type=date&logscale&legend=top-left)](https://www.star-history.com/?repos=alexlee-dev%2Fnano-vllm-labs&type=date&logscale=&legend=top-left)
