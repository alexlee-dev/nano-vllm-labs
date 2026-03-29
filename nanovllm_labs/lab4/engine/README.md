# Lab 4

The goal of Lab 4 is to keep the paged-KV runtime from Lab 3, then push it toward a more practical single-GPU serving engine.

This lab focuses on the next layer of runtime optimization:
- keep the prefix-cache and recompute-style preemption ideas from Lab 3
- size KV cache after a warmup pass instead of guessing from raw free memory
- capture CUDA Graphs for steady-state decode batches

Lab 4 is still intentionally scoped to one GPU, but it starts to resemble the execution strategy of a real serving runtime rather than only a teaching demo.

## What You Will Build

From the API point of view, Lab 4 still exposes the same top-level interface:

```python
llm = LLMEngine(model_path)
outputs = llm.generate(
    ["introduce yourself", "list all prime numbers within 100"],
    SamplingParams(temperature=0.6, max_tokens=128),
)
```

What changes is the execution behavior underneath:
1. the Lab 3 prefix-cache path is preserved, so repeated prompt blocks can still skip redundant prefill work
2. the Lab 3 recompute-style preemption path is preserved, so decode can still free blocks and retry when cache pressure blocks append
3. the runner warms up once before reserving KV cache capacity
4. decode batches can replay pre-captured CUDA Graphs instead of launching the eager path every step

## What Changes From Lab 3

Lab 3 introduced paged KV cache, prefill/decode scheduling, continuous batching, and in this codebase already included basic prefix reuse plus recompute-style preemption. Lab 4 keeps that structure, but attacks the next bottlenecks that show up on a single GPU:

- raw `cudaMemGetInfo()` is not enough once warmup allocations and graph buffers matter
- eager kernel launch overhead becomes noticeable in steady-state decode
- runtime status becomes easier to reason about when waiting, running, and finished states are explicit

This means Lab 4 is less about changing the public API and more about improving the engine's execution policy for realistic serving workloads.

## Suggested Module Split

### 1. `Sequence`

Responsibility: store request state plus runtime status.

Compared with Lab 3, the sequence now also tracks whether it is:
- `WAITING`
- `RUNNING`
- `FINISHED`

That extra state matters because preemption is now part of the runtime behavior.

### 2. `BlockManager`

Responsibility: carry forward paged KV cache allocation plus prefix cache reuse.

In this repository, the Lab 3 solution already hashes full blocks together with their prefix chain so cached prompt blocks can be reused across requests. Lab 4 keeps that behavior because the later decode optimizations still depend on the same block-table and cached-token accounting.

That preserved behavior still matters because:
- identical prompt prefixes can skip part of prefill
- KV blocks can behave like a small content-addressed cache instead of only a raw allocator

### 3. `Scheduler`

Responsibility: keep the engine making progress under cache pressure with clearer runtime state.

The scheduler still separates prefill and decode. As in the Lab 3 solution, it may preempt work when decode cannot append another token. Lab 4 keeps that behavior, but it now also tracks sequence status explicitly so the runtime state is less implicit.

Concretely, it has to decide when to:
- admit waiting requests whose uncached prompt tokens fit the batch budget
- keep decode moving when a sequence wants to append one more token
- preempt another running sequence to free blocks for forward progress

This is the point where the runtime starts acting more like a serving system and less like a simple batched loop.

### 4. `ModelRunner`

Responsibility: prepare tensors, reserve memory, and choose eager vs graph replay execution.

This is the main place where Lab 4 materially differs from Lab 3. The runner now:
- warm up the model once with a representative prefill batch
- estimate usable KV cache blocks from post-warmup memory stats
- keep reusable graph input/output buffers for decode
- replay a captured CUDA Graph for supported decode batch sizes

The main idea is that decode becomes a specialized fast path rather than just "the other branch" of the same eager forward call.

### 5. Cache-Aware Attention

Responsibility: stay compatible with both prefix cache reuse and decode graph replay.

Attention still reads and writes paged KV cache, but Lab 4 relies more heavily on the runtime context being stable enough for:
- cached-prefix prefill
- one-token decode against existing cache blocks
- static-shaped graph capture and replay

## End-to-End Data Flow

The key Lab 4 runtime path is:

```text
prompt(s)
  -> tokenizer / token ids
  -> Sequence objects
  -> waiting queue
  -> Scheduler.schedule()
      -> prefill batch, still able to reuse cached prefixes
      -> or decode batch, still able to preempt under cache pressure
  -> ModelRunner.prepare_*
  -> warmup-sized KV cache / runtime context
  -> eager prefill or CUDA Graph decode replay
  -> logits
  -> sampling
  -> Scheduler.postprocess()
      -> append token
      -> free blocks or keep running
  -> repeat until all requests finish
```

The most important new mental model is that Lab 4 optimizes the steady-state serving path. The model architecture is mostly unchanged, but the runtime is now doing more work to avoid redundant compute and launch overhead.

## Recommended Implementation Order

If you are filling in the skeleton from scratch, the cleanest order is:

1. Extend `Sequence`
   - add runtime status tracking

2. Upgrade `BlockManager`
   - keep the Lab 3 prefix-cache behavior intact
   - keep reference counts correct under reuse

3. Upgrade the scheduler
   - preserve the Lab 3 prefill/decode split
   - preserve recompute-style preemption when decode cannot append
   - make sequence state explicit

4. Upgrade `ModelRunner`
   - add warmup
   - compute KV cache capacity from post-warmup memory stats
   - split eager execution from decode fast path

5. Add CUDA Graph capture
   - choose decode batch-size buckets
   - capture static buffers
   - replay for supported decode batches

6. Reconnect everything in `LLMEngine.generate`
   - enqueue requests
   - schedule repeatedly
   - run prefill or decode
   - postprocess finished sequences

## What Lab 4 Still Does Not Solve

Even after Lab 4 works, it is still not the full nano-vllm runtime:
- the lab is still single-GPU only
- distributed execution is out of scope
- production observability and async serving layers are out of scope
- the sampling stack is still intentionally simple

That is fine for this repository. The goal here is to make the core runtime ideas understandable and runnable on one GPU. For the fuller implementation beyond this lab scope, refer to nano-vllm directly.

## How To Run It

Run the student version:

```bash
make run-lab4
```

Run the reference solution:

```bash
make run-lab4-s
```

Run the benchmark:

```bash
make bench-lab4
make bench-lab4-s
```

## A Simple Mental Model

You can summarize Lab 4 in three sentences:

- Lab 3 built the paged-KV serving loop
- Lab 4 makes that loop more efficient on a single GPU
- warmup-based KV sizing plus CUDA Graph decode replay is the main new optimization theme
