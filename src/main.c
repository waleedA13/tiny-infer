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

    // int8: going to int8 and back should be off by half a step at most, and
    // matmul_q8 should land close to matmul.
    enum { TEST_IN_DIM = 2 * Q8_GROUP, TEST_OUT_DIM = 3, TEST_WEIGHT_COUNT = TEST_IN_DIM * TEST_OUT_DIM };
    float float_weights[TEST_WEIGHT_COUNT], float_input[TEST_IN_DIM];
    float round_trip_weights[TEST_WEIGHT_COUNT];
    float expected_out[TEST_OUT_DIM], int8_out[TEST_OUT_DIM];
    int8_t int8_weights[TEST_WEIGHT_COUNT], int8_input[TEST_IN_DIM];
    float weight_scales[TEST_WEIGHT_COUNT / Q8_GROUP], input_scales[TEST_IN_DIM / Q8_GROUP];
    for (int i = 0; i < TEST_WEIGHT_COUNT; i++) float_weights[i] = sinf(i * 0.37f) * (i % 7 + 1);
    for (int i = 0; i < TEST_IN_DIM; i++) float_input[i] = cosf(i * 0.11f);
    quantize_q8(int8_weights, weight_scales, float_weights, TEST_WEIGHT_COUNT);
    dequantize_q8(round_trip_weights, int8_weights, weight_scales, TEST_WEIGHT_COUNT);
    int round_trip_ok = 1;
    for (int i = 0; i < TEST_WEIGHT_COUNT; i++)
        if (fabsf(round_trip_weights[i] - float_weights[i]) > weight_scales[i / Q8_GROUP] * 0.5f + 1e-6f)
            round_trip_ok = 0;
    if (!round_trip_ok) { printf("FAIL quantize_q8 round trip\n"); fails++; }
    quantize_q8(int8_input, input_scales, float_input, TEST_IN_DIM);
    matmul(expected_out, float_input, float_weights, TEST_IN_DIM, TEST_OUT_DIM);
    matmul_q8(int8_out, int8_input, input_scales, int8_weights, weight_scales, TEST_IN_DIM, TEST_OUT_DIM);
    for (int i = 0; i < TEST_OUT_DIM; i++)
        if (fabsf(int8_out[i] - expected_out[i]) > 0.02f * (fabsf(expected_out[i]) + 1.0f)) {
            printf("FAIL matmul_q8 (row %d: %f vs %f)\n", i, int8_out[i], expected_out[i]); fails++; break;
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
    int use_int8 = 0, run_compare = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--int8")) use_int8 = 1;
        else if (!strcmp(argv[i], "--compare")) run_compare = 1;
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
    if (use_int8 && model_quantize(&m) != 0) return 1;

    Tokenizer t;
    if (tokenizer_load(&t, tok_path, c->vocab_size) != 0) return 1;
    if (run_compare) {
        int compare_status = compare_q8(model_path, &t, prompt, steps);
        tokenizer_free(&t);
        model_free(&m);
        return compare_status;
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
