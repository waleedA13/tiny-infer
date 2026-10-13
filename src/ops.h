#ifndef OPS_H
#define OPS_H

// The math kernels. These are where all the speed work happens:
// naive -> pthreads -> SIMD (NEON/AVX2) -> int8.

// out = rmsnorm(x) * weight, length n
void rmsnorm(float *out, const float *x, const float *weight, int n);

// In-place softmax over n values
void softmax(float *x, int n);

// Start the matmul thread pool. n_threads <= 0 means one per CPU core.
// Returns the number of threads in use, including the caller, or -1 on error.
// Without ops_init, matmul runs single-threaded.
int  ops_init(int n_threads);
void ops_shutdown(void);

// out(d) = W(d, n) @ x(n). W is row-major. Rows are split across the pool.
// Not reentrant: call it from one thread at a time.
void matmul(float *out, const float *x, const float *w, int n, int d);

#endif
