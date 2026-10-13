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
// values plus one float scale, so a float is roughly its int8 value * scale. As long
// as a row's length is a multiple of Q8_GROUP, a group never spills into the next row.
#define Q8_GROUP 32

// scale = biggest |value| in the group / 127. count has to be a multiple of Q8_GROUP.
void quantize_q8(int8_t *int8_values, float *group_scales, const float *float_values, long count);
void dequantize_q8(float *float_values, const int8_t *int8_values, const float *group_scales, long count);

// Same as matmul, but the weights and the input are both int8 with a scale per group.
// Each group's dot product adds up in int32, then gets scaled back to a float.
void matmul_q8(float *out, const int8_t *input_values, const float *input_scales,
               const int8_t *weight_values, const float *weight_scales, int in_dim, int out_dim);

#endif
