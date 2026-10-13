#ifndef OPS_H
#define OPS_H

// The math kernels. These are where all the speed work happens:
// naive -> pthreads -> SIMD (NEON/AVX2) -> int8.

// output = rmsnorm(input) * weight, all of length size. output can be the same buffer as input.
void rmsnorm(float *output, const float *input, const float *weight, int size);

// In-place softmax over size values
void softmax(float *values, int size);

// Starts the thread pool for matmul. Pass 0 to use one thread per CPU core.
// Returns how many threads it's using (main thread included), or -1 if it failed.
// If you never call this, matmul just runs on one thread.
int  ops_init(int requested_threads);
void ops_shutdown(void);

// output(out_size) = weights(out_size, in_size) @ input(in_size). weights is row-major.
// The rows get split across threads. Only call it from one thread at a time.
void matmul(float *output, const float *input, const float *weights, int in_size, int out_size);

#endif
