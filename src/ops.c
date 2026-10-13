#include "ops.h"

#include <math.h>

void rmsnorm(float *out, const float *x, const float *weight, int n) {
    // ss = mean(x^2); scale = 1 / sqrt(ss + 1e-5). out can be the same buffer as x.
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = weight[i] * x[i] * scale;
}

void softmax(float *x, int n) {
    // Subtract the max first so exp doesn't blow up, then make it sum to 1.
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
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
