# Lab 7 Solution

`lab7_solution` packages a minimal single-node pipeline-parallel runtime for this repo.

It intentionally reuses as much of the earlier labs as possible:

- the Lab 4 scheduler, paged KV cache, and continuous batching model
- the Lab 5 multi-process single-node rank startup pattern
- a stage-local model partition that keeps original layer names so safetensor loading still works

This is pipeline parallelism, not tensor parallelism or data parallelism:

- each GPU owns a contiguous slice of decoder layers
- hidden states and residuals are passed from one stage to the next
- only the last stage computes logits and samples the next token

The current Lab 7 scope is intentionally small:

- single node only
- eager execution only when `pipeline_parallel_size > 1`
- no TP/DP mixing
- no inter-stage overlap or microbatch pipeline scheduling

## Run

`lab7_solution` defaults to `Qwen3-4B`, since that is the model this PP path is meant to demonstrate.

Two GPUs:

```bash
.venv/bin/python example.py --lab 7 --solution --pipeline-parallel-size 2
.venv/bin/python bench.py --lab 7 --solution --pipeline-parallel-size 2
make run-lab7-s
make bench-lab7-s
```

If you want to override the model path explicitly:

```bash
.venv/bin/python example.py --lab 7 --solution --model ~/huggingface/Qwen3-4B/ --pipeline-parallel-size 2
.venv/bin/python bench.py --lab 7 --solution --model ~/huggingface/Qwen3-4B/ --pipeline-parallel-size 2
```

You can also run the stage-aware model on one GPU:

```bash
.venv/bin/python example.py --lab 7 --solution --pipeline-parallel-size 1
```

That mode is mainly useful as a correctness check for the stage-partitioned code path.

## Design Notes

Compared with Lab 4, the biggest changes are:

- `models/qwen3.py`: stage-aware model construction, with per-stage layer ownership
- `engine/model_runner.py`: NCCL process group setup plus inter-stage `send`/`recv`
- `utils/loader.py`: tolerant weight loading for stage-local parameter sets, including tied embedding fallback for checkpoints that omit `lm_head.weight`

The most important practical caveat is the tied-embedding case:

- some checkpoints store `model.embed_tokens.weight` but not `lm_head.weight`
- in PP, the embedding lives on the first stage while the LM head lives on the last stage
- Lab 7 therefore copies the embedding weights into the last-stage LM head when needed

Without that fix, first-token predictions can be completely wrong even though the pipeline itself appears to run successfully.
