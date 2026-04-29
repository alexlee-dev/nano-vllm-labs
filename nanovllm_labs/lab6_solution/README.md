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

By default, `lab6_solution` follows the Lab 4 model choice and uses `Qwen3-0.6B` unless you pass `--model` explicitly.

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

## When To Use DP vs TP

Replicated data parallelism and tensor parallelism solve different problems:

- Use Lab 6 DP when the model already fits on each GPU and you want higher throughput by serving independent requests on multiple replicas.
- Use Lab 5 TP when the model is too large for one GPU or when you need to shard model weights across devices to make the run feasible.

The practical tradeoff is:

- DP keeps the forward path local to each rank, so it avoids TP collectives on every token step.
- TP pays communication cost during the forward pass, but it lets you run larger dense models that would not fit as full replicas.

That is why Lab 6 is usually the better comparison against Lab 4 on `Qwen3-0.6B`, while Lab 5 is the right tool for `Qwen3-4B` when single-GPU memory pressure becomes the dominant constraint.
