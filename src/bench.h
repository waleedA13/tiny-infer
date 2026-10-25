#ifndef BENCH_H
#define BENCH_H

// Benchmark mode. Same thing every time: start from BOS, always pick the top token,
// run a fixed number of steps. One warm-up run first, then report the middle of the
// timed runs. No tokenizer and no printing while the clock is running.
int run_benchmark(const char *model_path, int num_steps, int num_runs, int thread_count);

#endif
