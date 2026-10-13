#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

// File format (tokenizer.bin): int max_token_length, then for each token:
// float score, int len, char bytes[len].
int tokenizer_load(Tokenizer *t, const char *path, int vocab_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->scores = malloc(vocab_size * sizeof(float));
    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) goto bad;
    for (int i = 0; i < vocab_size; i++) {
        int len;
        if (fread(&t->scores[i], sizeof(float), 1, f) != 1) goto bad;
        if (fread(&len, sizeof(int), 1, f) != 1) goto bad;
        t->vocab[i] = malloc(len + 1);
        if (fread(t->vocab[i], 1, len, f) != (size_t)len) goto bad;
        t->vocab[i][len] = '\0';
    }
    fclose(f);
    return 0;
bad:
    fprintf(stderr, "bad tokenizer file: %s\n", path);
    fclose(f);
    return -1;
}

void tokenizer_free(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
}

const char *tokenizer_decode(const Tokenizer *t, int prev_token, int token) {
    const char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;  // BOS is token 1
    return piece;
}

int tokenizer_encode(const Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    // TODO: real BPE. For now just start with BOS so generation runs unconditioned.
    (void)t; (void)text;
    if (max_tokens < 1) return 0;
    tokens[0] = 1;
    return 1;
}
