#ifndef SAMPLER_H
#define SAMPLER_H

// Index of the biggest value.
int argmax(const float *values, int count);

// temperature == 0 means greedy (argmax).
int sample(float *logits, int vocab_size, float temperature, unsigned long long *rng_state);

#endif
