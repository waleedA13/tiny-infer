#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIRST_BYTE_TOKEN 3  // tokens <0x00>..<0xFF> start at this id

// File format (tokenizer.bin): int max_token_length, then for each token:
// float score, int length, char bytes[length].
int tokenizer_load(Tokenizer *tokenizer, const char *path, int vocab_size) {
    FILE *file = fopen(path, "rb");
    if (!file) { perror(path); return -1; }
    tokenizer->vocab_size = vocab_size;
    tokenizer->vocab = malloc(vocab_size * sizeof(char *));
    tokenizer->scores = malloc(vocab_size * sizeof(float));
    if (fread(&tokenizer->max_token_length, sizeof(int), 1, file) != 1) goto bad_file;
    for (int token_id = 0; token_id < vocab_size; token_id++) {
        int piece_length;
        if (fread(&tokenizer->scores[token_id], sizeof(float), 1, file) != 1) goto bad_file;
        if (fread(&piece_length, sizeof(int), 1, file) != 1) goto bad_file;
        tokenizer->vocab[token_id] = malloc(piece_length + 1);
        if (fread(tokenizer->vocab[token_id], 1, piece_length, file) != (size_t)piece_length) goto bad_file;
        tokenizer->vocab[token_id][piece_length] = '\0';
    }
    fclose(file);
    return 0;
bad_file:
    fprintf(stderr, "bad tokenizer file: %s\n", path);
    fclose(file);
    return -1;
}

void tokenizer_free(Tokenizer *tokenizer) {
    for (int token_id = 0; token_id < tokenizer->vocab_size; token_id++) free(tokenizer->vocab[token_id]);
    free(tokenizer->vocab);
    free(tokenizer->scores);
}

const char *tokenizer_decode(const Tokenizer *tokenizer, int previous_token, int token) {
    const char *piece = tokenizer->vocab[token];
    if (previous_token == BOS_TOKEN && piece[0] == ' ') piece++;
    return piece;
}

typedef struct {
    const char *piece;
    int token_id;
} SortedPiece;

static int compare_pieces(const void *left, const void *right) {
    return strcmp(((const SortedPiece *)left)->piece, ((const SortedPiece *)right)->piece);
}

static int find_token(const SortedPiece *sorted_vocab, int vocab_size, const char *piece) {
    SortedPiece search_key = { piece, 0 };
    const SortedPiece *match = bsearch(&search_key, sorted_vocab, vocab_size, sizeof(SortedPiece), compare_pieces);
    return match ? match->token_id : -1;
}

// BPE, the same way the TinyStories tokenizer does it:
// start with BOS and a space, then one token per character. If a character
// isn't in the vocab, use its raw bytes instead (tokens <0x00>..<0xFF>, ids 3..258).
// Then keep merging the neighbor pair with the best score until nothing merges.
int tokenizer_encode(const Tokenizer *tokenizer, const char *text, int *tokens, int max_tokens) {
    if (max_tokens < 1) return 0;

    int vocab_size = tokenizer->vocab_size;
    SortedPiece *sorted_vocab = malloc(vocab_size * sizeof(SortedPiece));
    size_t text_length = strlen(text);
    int *work_tokens = malloc((text_length + 2) * sizeof(int));  // BOS + space + one per byte
    char *merged_piece = malloc(2 * (size_t)tokenizer->max_token_length + 1);
    if (!sorted_vocab || !work_tokens || !merged_piece) {
        free(sorted_vocab); free(work_tokens); free(merged_piece);
        tokens[0] = BOS_TOKEN;
        return 1;
    }
    for (int token_id = 0; token_id < vocab_size; token_id++)
        sorted_vocab[token_id] = (SortedPiece){ tokenizer->vocab[token_id], token_id };
    qsort(sorted_vocab, vocab_size, sizeof(SortedPiece), compare_pieces);

    int token_count = 0;
    work_tokens[token_count++] = BOS_TOKEN;
    if (text_length > 0) {
        int space_token = find_token(sorted_vocab, vocab_size, " ");
        if (space_token >= 0) work_tokens[token_count++] = space_token;
    }

    // One token per character. In UTF-8 a character is one lead byte plus up to
    // 3 bytes that look like 10xxxxxx.
    char character[5];
    for (size_t text_position = 0; text_position < text_length;) {
        size_t char_length = 1;
        while (char_length < 4 && text_position + char_length < text_length &&
               ((unsigned char)text[text_position + char_length] & 0xC0) == 0x80)
            char_length++;
        memcpy(character, text + text_position, char_length);
        character[char_length] = '\0';
        int token_id = find_token(sorted_vocab, vocab_size, character);
        if (token_id >= 0) {
            work_tokens[token_count++] = token_id;
        } else {
            for (size_t byte_index = 0; byte_index < char_length; byte_index++)
                work_tokens[token_count++] = (unsigned char)character[byte_index] + FIRST_BYTE_TOKEN;
        }
        text_position += char_length;
    }

    // Merge pairs. Start after BOS so it never gets merged with text.
    for (;;) {
        float best_score = -1e10f;
        int best_token = -1, best_position = -1;
        for (int i = 1; i + 1 < token_count; i++) {
            const char *left_piece = tokenizer->vocab[work_tokens[i]];
            const char *right_piece = tokenizer->vocab[work_tokens[i + 1]];
            size_t left_length = strlen(left_piece), right_length = strlen(right_piece);
            if (left_length + right_length > 2 * (size_t)tokenizer->max_token_length) continue;
            memcpy(merged_piece, left_piece, left_length);
            memcpy(merged_piece + left_length, right_piece, right_length + 1);
            int merged_token = find_token(sorted_vocab, vocab_size, merged_piece);
            if (merged_token >= 0 && tokenizer->scores[merged_token] > best_score) {
                best_score = tokenizer->scores[merged_token];
                best_token = merged_token;
                best_position = i;
            }
        }
        if (best_position < 0) break;
        work_tokens[best_position] = best_token;
        memmove(work_tokens + best_position + 1, work_tokens + best_position + 2,
                (token_count - best_position - 2) * sizeof(int));
        token_count--;
    }

    if (token_count > max_tokens) token_count = max_tokens;
    memcpy(tokens, work_tokens, token_count * sizeof(int));
    free(sorted_vocab); free(work_tokens); free(merged_piece);
    return token_count;
}
