#ifndef TOKENIZER_H
#define TOKENIZER_H

#define BOS_TOKEN 1  // also the end of a story when the model picks it

typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_length;
} Tokenizer;

int  tokenizer_load(Tokenizer *tokenizer, const char *path, int vocab_size);
void tokenizer_free(Tokenizer *tokenizer);

// Returns the text for a token id (previous_token is used to strip the leading space after BOS).
const char *tokenizer_decode(const Tokenizer *tokenizer, int previous_token, int token);

// Turns text into token ids, starting with BOS. Returns how many tokens it wrote.
int tokenizer_encode(const Tokenizer *tokenizer, const char *text, int *tokens, int max_tokens);

#endif
