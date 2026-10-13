#ifndef COMPARE_H
#define COMPARE_H

#include "tokenizer.h"

// Runs the same prompt through an fp32 and an int8 copy of the model and prints
// how far apart they are (logit error, KL, top-1 agreement), how fast each one is,
// and the greedy story each one writes.
int compare_q8(const char *model_path, const Tokenizer *t, const char *prompt, int steps);

#endif
