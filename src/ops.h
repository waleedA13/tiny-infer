#ifndef OPS_H
#define OPS_H

// The math kernels. These are where all the speed work happens:
// naive -> pthreads -> SIMD (NEON/AVX2) -> int8.

// out = rmsnorm(x) * weight, length n
void rmsnorm(float *out, const float *x, const float *weight, int n);

// In-place softmax over n values
void softmax(float *x, int n);

// out(d) = W(d, n) @ x(n). W is row-major.
void matmul(float *out, const float *x, const float *w, int n, int d);

#endif
