#define _POSIX_C_SOURCE 200809L  // clock_gettime under -std=c11

#include "model.h"
#include "ops.h"
#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Small checks for the math kernels. Add one every time you optimize a kernel,
// and compare the fast version against the naive one.
static int selftest(void) {
    int fails = 0;

    float w[6] = {1, 2, 3, 4, 5, 6}, x[3] = {1, 1, 1}, out[2];
    matmul(out, x, w, 3, 2);
    if (out[0] != 6 || out[1] != 15) { printf("FAIL matmul\n"); fails++; }

    // Threaded matmul must match the single-threaded one bit for bit, on the
    // stories15M shapes and on row counts that don't split evenly.
    int shapes[][2] = {{288, 288}, {288, 768}, {768, 288}, {288, 32000}, {7, 5}, {13, 3}};
    for (size_t k = 0; k < sizeof(shapes) / sizeof(shapes[0]); k++) {
        int n = shapes[k][0], d = shapes[k][1];
        float *W = malloc((size_t)n * d * sizeof(float)), *X = malloc(n * sizeof(float));
        float *ref = malloc(d * sizeof(float)), *got = malloc(d * sizeof(float));
        unsigned long long seed = 42;
        for (long i = 0; i < (long)n * d; i++) W[i] = (float)((seed = seed * 6364136223846793005ULL + 1) >> 40) / 16777216.0f - 0.5f;
        for (int i = 0; i < n; i++) X[i] = (float)((seed = seed * 6364136223846793005ULL + 1) >> 40) / 16777216.0f - 0.5f;
        matmul(ref, X, W, n, d);
        for (int threads = 2; threads <= 8; threads *= 2) {
            ops_init(threads);
            matmul(got, X, W, n, d);
            ops_shutdown();
            if (memcmp(ref, got, d * sizeof(float)) != 0) {
                printf("FAIL threaded matmul (n=%d d=%d threads=%d)\n", n, d, threads);
                fails++;
            }
        }
        free(W); free(X); free(ref); free(got);
    }

    float s[3] = {1, 2, 3};
    softmax(s, 3);
    float sum = s[0] + s[1] + s[2];
    if (fabsf(sum - 1.0f) > 1e-5f) { printf("FAIL softmax (sum = %f)\n", sum); fails++; }

    float v[2] = {3, 4}, ones[2] = {1, 1}, r[2];
    rmsnorm(r, v, ones, 2);
    float expect = 3.0f / sqrtf(12.5f + 1e-5f);
    if (fabsf(r[0] - expect) > 1e-4f) { printf("FAIL rmsnorm\n"); fails++; }

    printf(fails ? "%d test(s) failed\n" : "all tests passed\n", fails);
    return fails ? 1 : 0;
}

static void usage(void) {
    fprintf(stderr,
        "usage: tiny-infer <model.bin> [-z tokenizer.bin] [-n steps] [-t temperature] [-i prompt]\n"
        "                  [-s seed] [-j threads]\n"
        "       tiny-infer --selftest\n");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 2) { usage(); return 1; }

    const char *model_path = argv[1];
    const char *tok_path = "models/tokenizer.bin";
    const char *prompt = "";
    int steps = 256;
    float temperature = 0.8f;
    unsigned long long rng = (unsigned long long)time(NULL);
    int threads = 0;  // 0 = one per CPU core
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-z")) tok_path = argv[i + 1];
        else if (!strcmp(argv[i], "-n")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t")) temperature = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-i")) prompt = argv[i + 1];
        else if (!strcmp(argv[i], "-s")) rng = strtoull(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "-j")) threads = atoi(argv[i + 1]);
        else { usage(); return 1; }
    }

    Model m;
    if (model_load(&m, model_path) != 0) return 1;
    Config *c = &m.config;
    printf("dim=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
           c->dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);
    if (steps > c->seq_len) steps = c->seq_len;
    threads = ops_init(threads);
    if (threads < 0) return 1;
    printf("threads=%d\n", threads);

    Tokenizer t;
    if (tokenizer_load(&t, tok_path, c->vocab_size) != 0) return 1;

    int prompt_tokens[512];
    int n_prompt = tokenizer_encode(&t, prompt, prompt_tokens, 512);

    if (rng == 0) rng = 1;  // xorshift gets stuck at 0
    int token = prompt_tokens[0], next;
    double start = 0;

    for (int pos = 0; pos < steps; pos++) {
        float *logits = model_forward(&m, token, pos);
        if (pos < n_prompt - 1) next = prompt_tokens[pos + 1];   // still feeding the prompt
        else next = sample(logits, c->vocab_size, temperature, &rng);
        if (next == 1) break;                                      // BOS ends the story
        printf("%s", tokenizer_decode(&t, token, next));
        fflush(stdout);
        token = next;
        if (pos == 0) start = now_ms();                            // skip warm-up step
    }

    if (steps > 1) printf("\n\n%.1f tok/s\n", (steps - 1) / ((now_ms() - start) / 1000.0));

    ops_shutdown();
    tokenizer_free(&t);
    model_free(&m);
    return 0;
}
