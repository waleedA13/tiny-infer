#include "compare.h"
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

    // int8: round trip error is at most half a step, and matmul_q8 tracks matmul.
    enum { N = 2 * Q8_GROUP, D = 3 };
    float wf[D * N], xf[N], wd[D * N], ref[D], got[D];
    int8_t wq[D * N], xq[N];
    float ws[D * N / Q8_GROUP], xs[N / Q8_GROUP];
    for (int i = 0; i < D * N; i++) wf[i] = sinf(i * 0.37f) * (i % 7 + 1);
    for (int i = 0; i < N; i++) xf[i] = cosf(i * 0.11f);
    quantize_q8(wq, ws, wf, D * N);
    dequantize_q8(wd, wq, ws, D * N);
    int rt_ok = 1;
    for (int i = 0; i < D * N; i++)
        if (fabsf(wd[i] - wf[i]) > ws[i / Q8_GROUP] * 0.5f + 1e-6f) rt_ok = 0;
    if (!rt_ok) { printf("FAIL quantize_q8 round trip\n"); fails++; }
    quantize_q8(xq, xs, xf, N);
    matmul(ref, xf, wf, N, D);
    matmul_q8(got, xq, xs, wq, ws, N, D);
    for (int i = 0; i < D; i++)
        if (fabsf(got[i] - ref[i]) > 0.02f * (fabsf(ref[i]) + 1.0f)) {
            printf("FAIL matmul_q8 (row %d: %f vs %f)\n", i, got[i], ref[i]); fails++; break;
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
        "                  [--int8] [--compare]\n"
        "       tiny-infer --selftest\n"
        "  --int8     quantize the weights to int8 and run with matmul_q8\n"
        "  --compare  run fp32 and int8 side by side (greedy) and report error and speed\n");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 2) { usage(); return 1; }

    const char *model_path = argv[1];
    const char *tok_path = "models/tokenizer.bin";
    const char *prompt = "";
    int steps = 256;
    float temperature = 0.8f;
    int int8 = 0, compare = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--int8")) int8 = 1;
        else if (!strcmp(argv[i], "--compare")) compare = 1;
        else if (i + 1 >= argc) { usage(); return 1; }
        else if (!strcmp(argv[i], "-z")) tok_path = argv[++i];
        else if (!strcmp(argv[i], "-n")) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "-i")) prompt = argv[++i];
        else { usage(); return 1; }
    }

    Model m;
    if (model_load(&m, model_path) != 0) return 1;
    Config *c = &m.config;
    printf("dim=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
           c->dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);
    if (steps > c->seq_len) steps = c->seq_len;
    if (int8 && model_quantize(&m) != 0) return 1;

    Tokenizer t;
    if (tokenizer_load(&t, tok_path, c->vocab_size) != 0) return 1;
    if (compare) {
        int rc = compare_q8(model_path, &t, prompt, steps);
        tokenizer_free(&t);
        model_free(&m);
        return rc;
    }

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
        if (pos == 0) start = now_ms();                            // skip warm-up step
    }

    if (steps > 1) printf("\n\n%.1f tok/s\n", (steps - 1) / ((now_ms() - start) / 1000.0));

    tokenizer_free(&t);
    model_free(&m);
    return 0;
}
