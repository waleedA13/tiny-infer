#ifndef SAMPLER_H
#define SAMPLER_H

// temperature == 0 means greedy (argmax).
int sample(float *logits, int n, float temperature, unsigned long long *rng);

#endif
