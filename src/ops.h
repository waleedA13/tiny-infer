#ifndef OPS_H
#define OPS_H

#include <stdint.h>

// The math kernels. These are where all the speed work happens:
// naive -> pthreads -> SIMD (NEON/AVX2) -> int8.

// out = rmsnorm(x) * weight, length n
void rmsnorm(float *out, const float *x, const float *weight, int n);

// In-place softmax over n values
void softmax(float *x, int n);

// out(d) = W(d, n) @ x(n). W is row-major.
void matmul(float *out, const float *x, const float *w, int n, int d);

// ---- int8 quantization ----
// Floats get turned into int8 in groups of Q8_GROUP. Each group keeps its int8
// values plus one float scale, so x is roughly q * scale. As long as a row's length
// is a multiple of Q8_GROUP, a group never spills into the next row.
#define Q8_GROUP 32

// scale = biggest |x| in the group / 127. n has to be a multiple of Q8_GROUP.
void quantize_q8(int8_t *q, float *scale, const float *x, long n);
void dequantize_q8(float *x, const int8_t *q, const float *scale, long n);

// Same as matmul, but W and x are both int8 with a scale per group.
// Each group's dot product adds up in int32, then gets scaled back to a float.
void matmul_q8(float *out, const int8_t *xq, const float *xs,
               const int8_t *wq, const float *ws, int n, int d);

#endif
