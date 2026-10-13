#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

typedef struct {
    const char *str;
    int id;
} TokenIndex;

static int cmp_token(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

static int lookup(const TokenIndex *sorted, int n, const char *str) {
    TokenIndex key = { str, 0 };
    const TokenIndex *hit = bsearch(&key, sorted, n, sizeof(TokenIndex), cmp_token);
    return hit ? hit->id : -1;
}

// SentencePiece-style BPE, matching how the TinyStories tokenizer was trained:
// BOS, a dummy leading space, one token per UTF-8 codepoint (falling back to
// raw-byte tokens <0x00>..<0xFF> at ids 3..258), then repeatedly merge the
// adjacent pair whose merged string has the highest score.
int tokenizer_encode(const Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    if (max_tokens < 1) return 0;

    TokenIndex *sorted = malloc(t->vocab_size * sizeof(TokenIndex));
    size_t len = strlen(text);
    int *buf = malloc((len + 2) * sizeof(int));                 // BOS + space + one per byte
    char *pair = malloc(2 * (size_t)t->max_token_length + 1);
    if (!sorted || !buf || !pair) {
        free(sorted); free(buf); free(pair);
        tokens[0] = 1;
        return 1;
    }
    for (int i = 0; i < t->vocab_size; i++) sorted[i] = (TokenIndex){ t->vocab[i], i };
    qsort(sorted, t->vocab_size, sizeof(TokenIndex), cmp_token);

    int n = 0;
    buf[n++] = 1;  // BOS
    if (len > 0) {
        int space = lookup(sorted, t->vocab_size, " ");
        if (space >= 0) buf[n++] = space;
    }

    // One token per codepoint. A UTF-8 codepoint is a lead byte plus any
    // continuation bytes (10xxxxxx), at most 4 bytes in total.
    char cp[5];
    for (size_t i = 0; i < len;) {
        size_t cp_len = 1;
        while (cp_len < 4 && i + cp_len < len && ((unsigned char)text[i + cp_len] & 0xC0) == 0x80)
            cp_len++;
        memcpy(cp, text + i, cp_len);
        cp[cp_len] = '\0';
        int id = lookup(sorted, t->vocab_size, cp);
        if (id >= 0) buf[n++] = id;
        else for (size_t b = 0; b < cp_len; b++) buf[n++] = (unsigned char)cp[b] + 3;
        i += cp_len;
    }

    // Greedy merges, starting after BOS so it never merges with text.
    for (;;) {
        float best_score = -1e10f;
        int best_id = -1, best_at = -1;
        for (int i = 1; i + 1 < n; i++) {
            size_t a = strlen(t->vocab[buf[i]]), b = strlen(t->vocab[buf[i + 1]]);
            if (a + b > 2 * (size_t)t->max_token_length) continue;
            memcpy(pair, t->vocab[buf[i]], a);
            memcpy(pair + a, t->vocab[buf[i + 1]], b + 1);
            int id = lookup(sorted, t->vocab_size, pair);
            if (id >= 0 && t->scores[id] > best_score) {
                best_score = t->scores[id];
                best_id = id;
                best_at = i;
            }
        }
        if (best_at < 0) break;
        buf[best_at] = best_id;
        memmove(buf + best_at + 1, buf + best_at + 2, (n - best_at - 2) * sizeof(int));
        n--;
    }

    if (n > max_tokens) n = max_tokens;
    memcpy(tokens, buf, n * sizeof(int));
    free(sorted); free(buf); free(pair);
    return n;
}
