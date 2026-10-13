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
// Values are quantized in groups of Q8_GROUP consecutive floats. Each group stores
// int8 values plus one float scale, so x ~= q * scale. A group never crosses a
// matrix row as long as the row length is a multiple of Q8_GROUP.
#define Q8_GROUP 32

// Symmetric per-group quantization: scale = max|x| / 127. n must be a multiple of Q8_GROUP.
void quantize_q8(int8_t *q, float *scale, const float *x, long n);
void dequantize_q8(float *x, const int8_t *q, const float *scale, long n);

// Same shape as matmul, but both W and x are int8 with per-group scales.
// Each group's dot product accumulates in int32, then gets scaled back to float.
void matmul_q8(float *out, const int8_t *xq, const float *xs,
               const int8_t *wq, const float *ws, int n, int d);

#endif
