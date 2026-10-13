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

static int compare_doubles(const void *left_ptr, const void *right_ptr) {
    double left = *(const double *)left_ptr, right = *(const double *)right_ptr;
    return (left > right) - (left < right);
}

// Benchmark mode. Same thing every time: start from BOS, always pick the top token,
// run a fixed number of steps. One warm-up run first, then report the middle of the
// timed runs. No tokenizer and no printing while the clock is running.
static int run_benchmark(const char *model_path, int num_steps, int num_runs) {
    Model model;
    if (model_load(&model, model_path) != 0) return 1;
    Config *config = &model.config;
    if (num_steps > config->seq_len) num_steps = config->seq_len;
    if (num_runs < 1) num_runs = 1;

    printf("model    %s\n", model_path);
    printf("config   dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
           config->dim, config->hidden_dim, config->n_layers, config->n_heads,
           config->n_kv_heads, config->vocab_size, config->seq_len);
    printf("build    %s | %s\n", __VERSION__, BUILD_FLAGS);
    printf("workload %d tokens from BOS, greedy, 1 warm-up + %d timed runs\n\n", num_steps, num_runs);

    double *tokens_per_second = malloc(num_runs * sizeof(double));
    double total_ms_all_runs = 0, sampling_ms_all_runs = 0, op_ms_all_runs[PROF_N] = {0};
    for (int run_index = -1; run_index < num_runs; run_index++) {   // -1 is the warm-up
        unsigned long long rng_state = 42;
        double sampling_ms = 0;
        int current_token = 1;
        prof_reset();
        prof_enabled = 1;
        double run_start_ms = prof_now_ms();
        for (int position = 0; position < num_steps; position++) {
            float *logits = model_forward(&model, current_token, position);
            double sample_start_ms = prof_now_ms();
            current_token = sample(logits, config->vocab_size, 0.0f, &rng_state);
            sampling_ms += prof_now_ms() - sample_start_ms;
        }
        double run_ms = prof_now_ms() - run_start_ms;
        prof_enabled = 0;
        if (run_index < 0) continue;
        tokens_per_second[run_index] = num_steps / (run_ms / 1000.0);
        total_ms_all_runs += run_ms;
        sampling_ms_all_runs += sampling_ms;
        for (int op = 0; op < PROF_N; op++) op_ms_all_runs[op] += prof_ms[op];
    }

    double total_tokens = (double)num_runs * num_steps;
    double total_ms_per_token = total_ms_all_runs / total_tokens;
    double matmul_ms_per_token = op_ms_all_runs[PROF_MATMUL] / total_tokens;
    double attention_ms_per_token = op_ms_all_runs[PROF_ATTENTION] / total_tokens;
    double sampling_ms_per_token = sampling_ms_all_runs / total_tokens;
    double rest_ms_per_token = total_ms_per_token - matmul_ms_per_token
                             - attention_ms_per_token - sampling_ms_per_token;
    qsort(tokens_per_second, num_runs, sizeof(double), compare_doubles);

    printf("tok/s    %.1f median (min %.1f, max %.1f)\n\n",
           tokens_per_second[num_runs / 2], tokens_per_second[0], tokens_per_second[num_runs - 1]);
    printf("%-10s %10s %7s\n", "op", "ms/token", "share");
    printf("%-10s %10.4f %6.1f%%\n", "matmul", matmul_ms_per_token, 100 * matmul_ms_per_token / total_ms_per_token);
    printf("%-10s %10.4f %6.1f%%\n", "attention", attention_ms_per_token, 100 * attention_ms_per_token / total_ms_per_token);
    printf("%-10s %10.4f %6.1f%%\n", "sampling", sampling_ms_per_token, 100 * sampling_ms_per_token / total_ms_per_token);
    printf("%-10s %10.4f %6.1f%%\n", "rest", rest_ms_per_token, 100 * rest_ms_per_token / total_ms_per_token);
    printf("%-10s %10.4f %6.1f%%\n", "total", total_ms_per_token, 100.0);

    free(tokens_per_second);
    model_free(&model);
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
        int bench_steps = 200, bench_runs = 5;
        for (int i = 3; i + 1 < argc; i += 2) {
            if (!strcmp(argv[i], "-n")) bench_steps = atoi(argv[i + 1]);
            else if (!strcmp(argv[i], "-r")) bench_runs = atoi(argv[i + 1]);
            else { usage(); return 1; }
        }
        return run_benchmark(argv[2], bench_steps, bench_runs);
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
