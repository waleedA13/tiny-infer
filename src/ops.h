#ifndef OPS_H
#define OPS_H

// The math kernels. These are where all the speed work happens:
// naive -> pthreads -> SIMD (NEON/AVX2) -> int8.

// out = rmsnorm(x) * weight, length n
void rmsnorm(float *out, const float *x, const float *weight, int n);

// In-place softmax over n values
void softmax(float *x, int n);

// Starts the thread pool for matmul. Pass 0 to use one thread per CPU core.
// Returns how many threads it's using (main thread included), or -1 if it failed.
// If you never call this, matmul just runs on one thread.
int  ops_init(int requested_threads);
void ops_shutdown(void);

// out(d) = W(d, n) @ x(n). W is row-major. The rows get split across threads.
// Only call it from one thread at a time.
void matmul(float *out, const float *x, const float *w, int n, int d);

#endif
