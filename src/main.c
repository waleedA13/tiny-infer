#include "model.h"
#include "ops.h"
#include "prof.h"
#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown"
#endif

// Small checks for the math kernels. Add one every time you optimize a kernel,
// and compare the fast version against the naive one.
static int selftest(void) {
    int fails = 0;

    float w[6] = {1, 2, 3, 4, 5, 6}, x[3] = {1, 1, 1}, out[2];
    matmul(out, x, w, 3, 2);
    if (out[0] != 6 || out[1] != 15) { printf("FAIL matmul\n"); fails++; }

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

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// Benchmark mode. Same thing every time: start from BOS, always pick the top token,
// run a fixed number of steps. One warm-up run first, then report the middle of the
// timed runs. No tokenizer and no printing while the clock is running.
static int bench(const char *model_path, int steps, int runs) {
    Model m;
    if (model_load(&m, model_path) != 0) return 1;
    Config *c = &m.config;
    if (steps > c->seq_len) steps = c->seq_len;
    if (runs < 1) runs = 1;

    printf("model    %s\n", model_path);
    printf("config   dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);
    printf("build    %s | %s\n", __VERSION__, BUILD_FLAGS);
    printf("workload %d tokens from BOS, greedy, 1 warm-up + %d timed runs\n\n", steps, runs);

    double *tok_s = malloc(runs * sizeof(double));
    double sum_total = 0, sum_sample = 0, sum_op[PROF_N] = {0};
    for (int r = -1; r < runs; r++) {          // r == -1 is the warm-up
        unsigned long long rng = 42;
        double sample_ms = 0;
        int token = 1;
        prof_reset();
        prof_enabled = 1;
        double start = prof_now_ms();
        for (int pos = 0; pos < steps; pos++) {
            float *logits = model_forward(&m, token, pos);
            double t0 = prof_now_ms();
            token = sample(logits, c->vocab_size, 0.0f, &rng);
            sample_ms += prof_now_ms() - t0;
        }
        double total = prof_now_ms() - start;
        prof_enabled = 0;
        if (r < 0) continue;
        tok_s[r] = steps / (total / 1000.0);
        sum_total += total;
        sum_sample += sample_ms;
        for (int k = 0; k < PROF_N; k++) sum_op[k] += prof_ms[k];
    }

    double per_tok = sum_total / ((double)runs * steps);   // ms per token, averaged
    double matmul = sum_op[PROF_MATMUL] / ((double)runs * steps);
    double attn = sum_op[PROF_ATTENTION] / ((double)runs * steps);
    double samp = sum_sample / ((double)runs * steps);
    double rest = per_tok - matmul - attn - samp;
    qsort(tok_s, runs, sizeof(double), cmp_double);

    printf("tok/s    %.1f median (min %.1f, max %.1f)\n\n", tok_s[runs / 2], tok_s[0], tok_s[runs - 1]);
    printf("%-10s %10s %7s\n", "op", "ms/token", "share");
    printf("%-10s %10.4f %6.1f%%\n", "matmul", matmul, 100 * matmul / per_tok);
    printf("%-10s %10.4f %6.1f%%\n", "attention", attn, 100 * attn / per_tok);
    printf("%-10s %10.4f %6.1f%%\n", "sampling", samp, 100 * samp / per_tok);
    printf("%-10s %10.4f %6.1f%%\n", "rest", rest, 100 * rest / per_tok);
    printf("%-10s %10.4f %6.1f%%\n", "total", per_tok, 100.0);

    free(tok_s);
    model_free(&m);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
        "usage: tiny-infer <model.bin> [-z tokenizer.bin] [-n steps] [-t temperature] [-i prompt]\n"
        "       tiny-infer --bench <model.bin> [-n steps] [-r runs]\n"
        "       tiny-infer --selftest\n");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc >= 3 && strcmp(argv[1], "--bench") == 0) {
        int steps = 200, runs = 5;
        for (int i = 3; i + 1 < argc; i += 2) {
            if (!strcmp(argv[i], "-n")) steps = atoi(argv[i + 1]);
            else if (!strcmp(argv[i], "-r")) runs = atoi(argv[i + 1]);
            else { usage(); return 1; }
        }
        return bench(argv[2], steps, runs);
    }
    if (argc < 2) { usage(); return 1; }

    const char *model_path = argv[1];
    const char *tok_path = "models/tokenizer.bin";
    const char *prompt = "";
    int steps = 256;
    float temperature = 0.8f;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-z")) tok_path = argv[i + 1];
        else if (!strcmp(argv[i], "-n")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t")) temperature = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-i")) prompt = argv[i + 1];
        else { usage(); return 1; }
    }

    Model m;
    if (model_load(&m, model_path) != 0) return 1;
    Config *c = &m.config;
    printf("dim=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
           c->dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);
    if (steps > c->seq_len) steps = c->seq_len;

    Tokenizer t;
    if (tokenizer_load(&t, tok_path, c->vocab_size) != 0) return 1;

    int prompt_tokens[512];
    int n_prompt = tokenizer_encode(&t, prompt, prompt_tokens, 512);

    unsigned long long rng = (unsigned long long)time(NULL);
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
        if (pos == 0) start = prof_now_ms();                       // skip warm-up step
    }

    if (steps > 1) printf("\n\n%.1f tok/s\n", (steps - 1) / ((prof_now_ms() - start) / 1000.0));

    tokenizer_free(&t);
    model_free(&m);
    return 0;
}
