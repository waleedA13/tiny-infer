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

void quantize_q8(int8_t *q, float *scale, const float *x, long n) {
    for (long g = 0; g < n / Q8_GROUP; g++) {
        const float *xg = x + g * Q8_GROUP;
        float amax = 0.0f;
        for (int k = 0; k < Q8_GROUP; k++) {
            float a = fabsf(xg[k]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.0f;
        float inv = s > 0.0f ? 1.0f / s : 0.0f;
        for (int k = 0; k < Q8_GROUP; k++) q[g * Q8_GROUP + k] = (int8_t)roundf(xg[k] * inv);
        scale[g] = s;
    }
}

void dequantize_q8(float *x, const int8_t *q, const float *scale, long n) {
    for (long i = 0; i < n; i++) x[i] = q[i] * scale[i / Q8_GROUP];
}

void matmul_q8(float *out, const int8_t *xq, const float *xs,
               const int8_t *wq, const float *ws, int n, int d) {
    // Naive like matmul. The int32 inner loop is what a compiler (or later, hand
    // written NEON/AVX2) can turn into wide integer multiply-adds.
    int groups = n / Q8_GROUP;
    for (int i = 0; i < d; i++) {
        const int8_t *row = wq + (long)i * n;
        const float *rs = ws + (long)i * groups;
        float sum = 0.0f;
        for (int g = 0; g < groups; g++) {
            int32_t acc = 0;
            for (int k = 0; k < Q8_GROUP; k++)
                acc += (int32_t)row[g * Q8_GROUP + k] * (int32_t)xq[g * Q8_GROUP + k];
            sum += (float)acc * rs[g] * xs[g];
        }
        out[i] = sum;
    }
}
