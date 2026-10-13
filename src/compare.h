#ifndef COMPARE_H
#define COMPARE_H

#include "tokenizer.h"

// Runs the same prompt through an fp32 and an int8 copy of the model. Prints how
// different their outputs are, how fast each one is, and the story each one writes.
int compare_q8(const char *model_path, const Tokenizer *t, const char *prompt, int steps);

#endif
