#include "sampler.h"
#include "ops.h"

static float random_float(unsigned long long *rng_state) {
    // xorshift64*
    *rng_state ^= *rng_state >> 12;
    *rng_state ^= *rng_state << 25;
    *rng_state ^= *rng_state >> 27;
    return ((*rng_state * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

static int argmax(const float *values, int size) {
    int best_index = 0;
    for (int i = 1; i < size; i++) if (values[i] > values[best_index]) best_index = i;
    return best_index;
}

int sample(float *logits, int vocab_size, float temperature, unsigned long long *rng_state) {
    if (temperature == 0.0f) return argmax(logits, vocab_size);
    for (int i = 0; i < vocab_size; i++) logits[i] /= temperature;
    softmax(logits, vocab_size);
    float target = random_float(rng_state), cumulative_prob = 0.0f;
    for (int token_id = 0; token_id < vocab_size; token_id++) {
        cumulative_prob += logits[token_id];
        if (target < cumulative_prob) return token_id;
    }
    return vocab_size - 1;
    // TODO (later): top-p sampling.
}
