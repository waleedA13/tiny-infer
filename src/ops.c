#include "ops.h"

#include <math.h>

void rmsnorm(float *out, const float *x, const float *weight, int n) {
    // TODO: ss = mean(x^2); scale = 1 / sqrt(ss + 1e-5); out[i] = weight[i] * x[i] * scale
    for (int i = 0; i < n; i++) out[i] = x[i] * weight[i];
}

void softmax(float *x, int n) {
    // TODO: subtract the max first for numerical stability, then exp and normalize.
    (void)x; (void)n;
}

void matmul(float *out, const float *x, const float *w, int n, int d) {
    // Naive version. This is the hottest loop in the program: profile it, then
    // split rows across threads, then vectorize the inner loop.
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += w[(long)i * n + j] * x[j];
        out[i] = sum;
    }
}

void quantize_q8(int8_t *int8_values, float *group_scales, const float *float_values, long count) {
    for (long group = 0; group < count / Q8_GROUP; group++) {
        const float *group_floats = float_values + group * Q8_GROUP;
        float largest_magnitude = 0.0f;
        for (int k = 0; k < Q8_GROUP; k++) {
            float magnitude = fabsf(group_floats[k]);
            if (magnitude > largest_magnitude) largest_magnitude = magnitude;
        }
        float scale = largest_magnitude / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int k = 0; k < Q8_GROUP; k++)
            int8_values[group * Q8_GROUP + k] = (int8_t)roundf(group_floats[k] * inverse_scale);
        group_scales[group] = scale;
    }
}

void dequantize_q8(float *float_values, const int8_t *int8_values, const float *group_scales, long count) {
    for (long i = 0; i < count; i++) float_values[i] = int8_values[i] * group_scales[i / Q8_GROUP];
}

void matmul_q8(float *out, const int8_t *input_values, const float *input_scales,
               const int8_t *weight_values, const float *weight_scales, int in_dim, int out_dim) {
    // Naive, same as matmul. The int32 inner loop is the part the compiler (or my
    // own NEON/AVX2 code later) can turn into wide integer multiply-adds.
    int groups_per_row = in_dim / Q8_GROUP;
    for (int i = 0; i < out_dim; i++) {
        const int8_t *row_values = weight_values + (long)i * in_dim;
        const float *row_scales = weight_scales + (long)i * groups_per_row;
        float row_sum = 0.0f;
        for (int group = 0; group < groups_per_row; group++) {
            int32_t group_dot = 0;
            for (int k = 0; k < Q8_GROUP; k++)
                group_dot += (int32_t)row_values[group * Q8_GROUP + k] *
                             (int32_t)input_values[group * Q8_GROUP + k];
            row_sum += (float)group_dot * row_scales[group] * input_scales[group];
        }
        out[i] = row_sum;
    }
}
