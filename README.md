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
./tiny-infer models/stories15M.bin -n 200 -t 0.8 --int8     # int8 weights and matmul
./tiny-infer models/stories15M.bin -n 200 --compare         # fp32 vs int8: error, speed, both stories
```

`--int8` quantizes every weight in groups of 32 (one float scale per group, `scale = max|w| / 127`).
Activations are quantized the same way right before each matmul, so the inner loop is an
int8 x int8 dot product accumulated in int32.

## Benchmarks

I'll fill this in as I optimize. Same model, same prompt, 200 tokens.

| Version | tok/s | Notes |
|---|---|---|
| naive | | |
| pthreads | | |
| NEON SIMD | | |
| int8 | | |

## Layout

```
src/main.c        CLI, generation loop, timing, self-tests
src/model.c/.h    checkpoint loading and the forward pass
src/ops.c/.h      rmsnorm, softmax, matmul (all the speed work goes here)
src/tokenizer.c/.h
src/sampler.c/.h
```

## Credit

The checkpoint and tokenizer formats come from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), and I used it as a reference for the file layout. The kernels, threading, SIMD, quantization, and MoE work in this repo are my own.
