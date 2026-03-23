# Lab 1

The goal of Lab 1 is to build a minimal but working LLM inference engine.

This lab is not about throughput, advanced scheduling, or a full vLLM-style runtime. The only real objective is to get the main generation path working end to end: given a prompt, generate text token by token.

In this repository, Lab 1 should be understood as a minimal baseline:
- Single-request generation
- Synchronous execution
- Full forward pass at every decoding step
- No KV cache
- No batching
- No continuous batching
- No memory management

## What You Will Build

From the API point of view, Lab 1 should support an interface like this:

```python
llm = LLMEngine(model_path)
outputs = llm.generate(
    ["introduce yourself"],
    SamplingParams(temperature=0.6, max_tokens=32),
)
```

Under the hood, that means four things:
1. Load the model and tokenizer
2. Convert the prompt into token ids
3. Repeatedly run the model and sample one new token each step
4. Decode the generated token ids back into text

## Suggested Module Split


### 1. `Sequence`

Responsibility: store the state of one request while it is being generated.

The most important fields are:
- `token_ids`
- `last_token`
- `num_tokens`
- `num_prompt_tokens`
- `temperature`
- `max_tokens`
- `ignore_eos`

The shared sequence state is already defined in base_sequence.py, so the Lab 1 `Sequence` can stay extremely thin. It can even be only a semantic wrapper.

Why use a `Sequence` object instead of passing around a plain `list[int]`?

Because every later lab will attach more request-level state:
- Finished status
- Block table
- KV cache locations
- Scheduling metadata
- Sampling state

So Lab 1 establishes an important idea early: a request is a first-class runtime object.

### 2. `ModelRunner`

Responsibility: run the model, and only run the model.

In model_runner.py, its responsibilities are intentionally narrow:
- Choose `cuda` or `cpu` from `device`
- Choose `float16` or `float32` from `dtype`
- Load the model with `transformers.AutoModelForCausalLM`
- Run one forward pass in `step()`
- Sample the next token from the final-position logits
- Append that token back to the sequence

That split matters because:
- `LLMEngine` does not need to know any `torch` details
- later changes like batching, KV cache, or paged attention mostly belong in the runner
- the API layer stays decoupled from the execution layer

### 3. `LLMEngine`

Responsibility: orchestrate the inference flow.

It does not implement the neural network itself. Instead, it ties the whole process together:
- Initialize the tokenizer and model runner
- Accept either string prompts or token ids
- Create sequences
- Repeatedly call `step()`
- Stop on EOS or when `max_tokens` is reached
- Return the final text and generated token ids

This is the core role of the engine layer: coordinate the workflow rather than owning every low-level detail.

## End-to-End Data Flow

The most important thing to fully understand in Lab 1 is this chain:

```text
prompt(string)
  -> tokenizer.encode(...)
  -> prompt_token_ids
  -> Sequence(prompt_token_ids, sampling_params)
  -> ModelRunner.step(seq)
  -> model forward
  -> next-token logits
  -> sampling
  -> seq.append_token(token_id)
  -> repeat
  -> tokenizer.decode(seq.completion_token_ids)
  -> final text
```

Once this path is clear, many later optimizations become easier to reason about. They are usually just changes to one part of the same chain:
- reduce redundant forward computation
- handle multiple sequences at once
- store and read KV cache more efficiently
- use more advanced scheduling

## Recommended Implementation Order

If you are filling in the skeleton from scratch, the cleanest order is:

1. Implement `LLMEngine.__init__`
   - Load the tokenizer
   - Create the model runner
   - Record `eos_token_id`

2. Implement the logic that turns a prompt into a sequence
   - The input may be a `str`
   - It may also already be a `list[int]`

3. Implement `ModelRunner`
   - Load `AutoModelForCausalLM`
   - Handle `device` and `dtype`
   - Implement one forward pass plus sampling

4. Implement `generate`
   - Iterate over requests
   - Run the token generation loop
   - Stop on the configured conditions
   - Collect outputs

This is much easier to debug than pushing all logic into one large `generate()` method.

## Stopping Conditions

Lab 1 should support at least two stopping conditions:
- `max_tokens` has been reached
- the generated token is `eos_token_id` and `ignore_eos=False`

These are basic, but enough for a minimal working demo.

## Sampling Logic

In this implementation, sampling is also kept intentionally simple:
- scale logits by `temperature`
- apply `softmax`
- sample one token with `torch.multinomial`

That means Lab 1 does not yet cover:
- greedy decoding
- top-k
- top-p
- repetition penalty
- beam search

Those can be added later. At this stage, the important thing is to make the minimal logits-to-token loop work correctly.


## How To Run It

Run the student version:

```bash
make run-lab1
```

Run the reference solution:

```bash
make run-lab1-s
```

Run the benchmark:

```bash
make bench-lab1
make bench-lab1-s
```


## A Simple Mental Model

You can summarize Lab 1 in three sentences:

- `Sequence` stores request state
- `ModelRunner` runs the model
- `LLMEngine` ties the generation flow together

If those three layers stay clean, the later optimizations have a solid place to land.
