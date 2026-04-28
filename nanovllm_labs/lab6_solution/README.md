# Lab 6 Solution

`lab6_solution` packages the replicated data-parallel runtime from this branch as a standalone lab solution.

It keeps the Lab 4 single-GPU scheduler, paged KV cache, warmup-based KV sizing, and CUDA Graph decode path, then scales out by:

- replicating the full dense model on each GPU
- assigning requests across ranks with a simple load-aware policy
- letting each rank schedule and execute its own prefill/decode loop
- merging per-rank step timing back into the shared benchmark metrics

This is data parallelism, not tensor parallelism:

- every GPU holds a full model copy
- requests are distributed across replicas
- there is no layer sharding or cross-rank all-reduce on the forward path

## Run

Single GPU:

```bash
.venv/bin/python example.py --lab 6 --solution --data-parallel-size 1
.venv/bin/python bench.py --lab 6 --solution --data-parallel-size 1
```

Two GPUs:

```bash
.venv/bin/python example.py --lab 6 --solution --data-parallel-size 2
.venv/bin/python bench.py --lab 6 --solution --data-parallel-size 2
make run-lab6-s
make bench-lab6-s
```

If you want to benchmark a larger dense model, pass it explicitly:

```bash
.venv/bin/python bench.py --lab 6 --solution --model ~/huggingface/Qwen3-4B --data-parallel-size 2
```

`--tensor-parallel-size 2` is accepted as a compatibility alias, but `lab6_solution` does not implement tensor parallelism.
