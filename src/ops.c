#include "ops.h"

#include <math.h>

void rmsnorm(float *output, const float *input, const float *weight, int size) {
    // mean_square = mean(input^2); scale = 1 / sqrt(mean_square + 1e-5).
    float sum_of_squares = 0.0f;
    for (int i = 0; i < size; i++) sum_of_squares += input[i] * input[i];
    float scale = 1.0f / sqrtf(sum_of_squares / size + 1e-5f);
    for (int i = 0; i < size; i++) output[i] = weight[i] * input[i] * scale;
}

void softmax(float *values, int size) {
    // Subtract the max first so exp doesn't blow up, then make it sum to 1.
    float max_value = values[0];
    for (int i = 1; i < size; i++) if (values[i] > max_value) max_value = values[i];
    float total = 0.0f;
    for (int i = 0; i < size; i++) { values[i] = expf(values[i] - max_value); total += values[i]; }
    for (int i = 0; i < size; i++) values[i] /= total;
}

void matmul(float *output, const float *input, const float *weights, int in_size, int out_size) {
    // Naive version. This is the hottest loop in the program: profile it, then
    // split rows across threads, then vectorize the inner loop.
    for (int row = 0; row < out_size; row++) {
        const float *weight_row = weights + (long)row * in_size;
        float sum = 0.0f;
        for (int col = 0; col < in_size; col++) sum += weight_row[col] * input[col];
        output[row] = sum;
    }
}
