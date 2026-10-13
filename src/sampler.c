#include "sampler.h"
#include "ops.h"

static float rand_f32(unsigned long long *state) {
    // xorshift64*
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return ((*state * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

int sample(float *logits, int n, float temperature, unsigned long long *rng) {
    if (temperature == 0.0f) return argmax(logits, n);
    for (int i = 0; i < n; i++) logits[i] /= temperature;
    softmax(logits, n);
    float r = rand_f32(rng), cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return n - 1;
    // TODO (later): top-p sampling.
}
