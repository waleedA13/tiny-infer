#ifndef OPS_H
#define OPS_H

// The math kernels. These are where all the speed work happens:
// naive -> pthreads -> SIMD (NEON/AVX2) -> int8.

// output = rmsnorm(input) * weight, all of length size. output can be the same buffer as input.
void rmsnorm(float *output, const float *input, const float *weight, int size);

// In-place softmax over size values
void softmax(float *values, int size);

// output(out_size) = weights(out_size, in_size) @ input(in_size). weights is row-major.
void matmul(float *output, const float *input, const float *weights, int in_size, int out_size);

#endif
