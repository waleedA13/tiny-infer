# tiny-infer

A small transformer inference engine in plain C. It runs a ~15M parameter language model on a laptop CPU and writes short stories, one token at a time. No PyTorch, no BLAS, just C and the math.

I'm building this to understand what happens under the hood when a model generates text, and to practice the low-level work I care about: multithreading, SIMD, quantization, and memory. It also lets me test an idea from my research on Mixture-of-Experts models: if you can guess which expert runs next, you can load its weights early and keep the CPU busy.

## Status

- [x] mmap checkpoint loader
- [x] tokenizer loading and decoding
- [x] sampling (greedy and temperature)
- [x] naive matmul
- [ ] rmsnorm and softmax
- [ ] attention with RoPE and a KV cache
- [ ] SwiGLU feed-forward
- [ ] BPE prompt encoding
- [ ] pthreads matmul
- [ ] NEON / AVX2 matmul
- [ ] int8 quantization
- [ ] toy MoE layer with expert prefetching

## Build and run

```bash
make            # normal build
make fast       # -O3 -march=native
make debug      # address and undefined behavior sanitizers
make test       # kernel self-tests
```

Get the model weights (TinyStories, about 60 MB):

```bash
./scripts/get_model.sh
```

Then generate:

```bash
./tiny-infer models/stories15M.bin -n 200 -t 0.8
```

## Benchmarks

```bash
make bench                                    # real weights (run ./scripts/get_model.sh first)
make bench MODEL=models/synthetic15M.bin      # fake weights, same size, no download needed
```

`make bench` builds a separate `tiny-infer-bench` binary with `-O3 -march=native -ffast-math` (so I never accidentally time a debug build) and runs it. It generates 200 tokens starting from BOS, always picking the top token, throws away one warm-up run, and reports the middle result of 5 runs. It also shows where the time goes for each token:

- **matmul**: time spent inside `matmul()`
- **attention**: the attention part of `model_forward` (scores, softmax, mixing the values), not counting its matmuls
- **sampling**: picking the next token
- **rest**: everything else (embedding, rmsnorm, RoPE, SwiGLU, adding residuals)

The speed doesn't care what the weights actually are, so `models/synthetic15M.bin` (make it with `make models/synthetic15M.bin`) does the same amount of work as the real model. You can change `BENCH_TOKENS`, `BENCH_RUNS` or `BENCH_FLAGS` on the make command line.

Same model size, 200 tokens, default `make bench` settings.

| Version | tok/s | matmul | attention | sampling | rest | Notes |
|---|---|---|---|---|---|---|
| skeleton | 666 | 96.5% | 0.0% | 3.4% | 0.1% | Attention and FFN aren't written yet, so this is just the embedding and the final layer. Ran on a cloud VM (Intel Xeon 2.1 GHz, one thread, GCC 13.3) with fake weights. |
| naive | | | | | | |
| pthreads | | | | | | |
| NEON SIMD | | | | | | |
| int8 | | | | | | |

## Layout

```
src/main.c        CLI, generation loop, benchmark mode, self-tests
src/model.c/.h    checkpoint loading and the forward pass
src/ops.c/.h      rmsnorm, softmax, matmul (all the speed work goes here)
src/tokenizer.c/.h
src/sampler.c/.h
src/prof.c/.h     timers for --bench
scripts/make_synthetic_model.py   fake weights the same size as stories15M
```

## Credit

The checkpoint and tokenizer formats come from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), and I used it as a reference for the file layout. The kernels, threading, SIMD, quantization, and MoE work in this repo are my own.
