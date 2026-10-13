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
make bench MODEL=models/synthetic15M.bin      # random weights, same shapes, no download
```

`make bench` rebuilds a separate `tiny-infer-bench` binary with `-O3 -march=native -ffast-math`, then runs `tiny-infer --bench`: 200 tokens from BOS with greedy sampling, one untimed warm-up run, and the median of 5 timed runs. It prints tok/s and a per-token time breakdown:

- **matmul**: time inside `matmul()` (every projection, the FFN, and the classifier)
- **attention**: the attention core in `model_forward` (scores, softmax, weighted sum of values), not counting its matmuls
- **sampling**: picking the next token from the logits
- **rest**: everything else (embedding lookup, rmsnorm, RoPE, SwiGLU, residual adds)

Speed doesn't depend on the weight values, so `models/synthetic15M.bin` (`make models/synthetic15M.bin`) measures the same work as the real checkpoint. Override `BENCH_TOKENS`, `BENCH_RUNS` or `BENCH_FLAGS` on the make command line to change the workload.

Same model shape, 200 tokens, `make bench` defaults.

| Version | tok/s | matmul | attention | sampling | rest | Notes |
|---|---|---|---|---|---|---|
| skeleton | 666 | 96.5% | 0.0% | 3.4% | 0.1% | Only embedding, final rmsnorm and classifier run; attention and FFN are still TODO. Intel Xeon @ 2.1 GHz cloud VM, single thread, GCC 13.3, synthetic weights. |
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
src/prof.c/.h     per-op timers used by --bench
scripts/make_synthetic_model.py   random weights shaped like stories15M
```

## Credit

The checkpoint and tokenizer formats come from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), and I used it as a reference for the file layout. The kernels, threading, SIMD, quantization, and MoE work in this repo are my own.
