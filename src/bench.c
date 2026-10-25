#include "bench.h"
#include "model.h"
#include "ops.h"
#include "prof.h"
#include "sampler.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown"
#endif

static int compare_doubles(const void *left_ptr, const void *right_ptr) {
    double left = *(const double *)left_ptr, right = *(const double *)right_ptr;
    return (left > right) - (left < right);
}

int run_benchmark(const char *model_path, int num_steps, int num_runs, int thread_count) {
    Model model;
    if (model_load(&model, model_path) != 0) return 1;
    Config *config = &model.config;
    if (num_steps > config->max_seq_len) num_steps = config->max_seq_len;
    if (num_runs < 1) num_runs = 1;
    thread_count = ops_init(thread_count);
    if (thread_count < 0) return 1;

    printf("model    %s\n", model_path);
    printf("config   embed_dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d max_seq_len=%d\n",
           config->embed_dim, config->hidden_dim, config->num_layers, config->num_heads,
           config->num_kv_heads, config->vocab_size, config->max_seq_len);
    printf("build    %s | %s\n", __VERSION__, BUILD_FLAGS);
    printf("threads  %d\n", thread_count);
    printf("workload %d tokens from BOS, greedy, 1 warm-up + %d timed runs\n\n", num_steps, num_runs);

    double *tokens_per_second = malloc(num_runs * sizeof(double));
    double total_ms_all_runs = 0, sampling_ms_all_runs = 0, op_ms_all_runs[PROF_CATEGORY_COUNT] = {0};
    for (int run_index = -1; run_index < num_runs; run_index++) {   // -1 is the warm-up
        unsigned long long rng_state = 42;
        double sampling_ms = 0;
        int current_token = BOS_TOKEN;
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
        for (int op = 0; op < PROF_CATEGORY_COUNT; op++) op_ms_all_runs[op] += prof_ms[op];
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
    ops_shutdown();
    model_free(&model);
    return 0;
}
