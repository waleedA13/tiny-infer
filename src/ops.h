#ifndef OPS_H
#define OPS_H

#include <stdint.h>

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
