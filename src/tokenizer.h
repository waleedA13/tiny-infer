#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_length;
} Tokenizer;

int  tokenizer_load(Tokenizer *t, const char *path, int vocab_size);
void tokenizer_free(Tokenizer *t);

// Returns the text for a token id (prev_token is used to strip the leading space after BOS).
const char *tokenizer_decode(const Tokenizer *t, int prev_token, int token);

// TODO: BPE encode the prompt into token ids. Returns the number of tokens written.
int tokenizer_encode(const Tokenizer *t, const char *text, int *tokens, int max_tokens);

#endif
