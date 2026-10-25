#include "bench.h"
#include "compare.h"
#include "model.h"
#include "ops.h"
#include "prof.h"
#include "sampler.h"
#include "selftest.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(void) {
    fprintf(stderr,
        "usage: tiny-infer <model.bin> [-z tokenizer.bin] [-n steps] [-t temperature] [-i prompt]\n"
        "                  [-s seed] [-j threads] [--int8] [--compare]\n"
        "       tiny-infer --bench <model.bin> [-n steps] [-r runs] [-j threads]\n"
        "       tiny-infer --selftest\n"
        "  --int8     quantize the weights to int8 and run with matmul_q8\n"
        "  --compare  run fp32 and int8 side by side (greedy) and report error and speed\n");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return run_selftest();
    if (argc >= 3 && strcmp(argv[1], "--bench") == 0) {
        int bench_steps = 200, bench_runs = 5, bench_threads = 0;
        for (int arg_index = 3; arg_index + 1 < argc; arg_index += 2) {
            if (!strcmp(argv[arg_index], "-n")) bench_steps = atoi(argv[arg_index + 1]);
            else if (!strcmp(argv[arg_index], "-r")) bench_runs = atoi(argv[arg_index + 1]);
            else if (!strcmp(argv[arg_index], "-j")) bench_threads = atoi(argv[arg_index + 1]);
            else { usage(); return 1; }
        }
        return run_benchmark(argv[2], bench_steps, bench_runs, bench_threads);
    }
    if (argc < 2) { usage(); return 1; }

    const char *model_path = argv[1];
    const char *tokenizer_path = "models/tokenizer.bin";
    const char *prompt = "";
    int steps = 256;
    float temperature = 0.8f;
    unsigned long long rng_state = (unsigned long long)time(NULL);
    int thread_count = 0;  // 0 = one per CPU core
    int use_int8 = 0, run_compare = 0;
    for (int arg_index = 2; arg_index < argc; arg_index++) {
        const char *flag = argv[arg_index];
        if (!strcmp(flag, "--int8")) { use_int8 = 1; continue; }
        if (!strcmp(flag, "--compare")) { run_compare = 1; continue; }
        if (arg_index + 1 >= argc) { usage(); return 1; }
        const char *value = argv[++arg_index];
        if (!strcmp(flag, "-z")) tokenizer_path = value;
        else if (!strcmp(flag, "-n")) steps = atoi(value);
        else if (!strcmp(flag, "-t")) temperature = atof(value);
        else if (!strcmp(flag, "-i")) prompt = value;
        else if (!strcmp(flag, "-s")) rng_state = strtoull(value, NULL, 10);
        else if (!strcmp(flag, "-j")) thread_count = atoi(value);
        else { usage(); return 1; }
    }

    Model model;
    if (model_load(&model, model_path) != 0) return 1;
    Config *config = &model.config;
    printf("embed_dim=%d layers=%d heads=%d kv_heads=%d vocab=%d max_seq_len=%d\n",
           config->embed_dim, config->num_layers, config->num_heads, config->num_kv_heads,
           config->vocab_size, config->max_seq_len);
    if (steps > config->max_seq_len) steps = config->max_seq_len;
    if (use_int8 && model_quantize(&model) != 0) return 1;
    thread_count = ops_init(thread_count);
    if (thread_count < 0) return 1;
    printf("threads=%d\n", thread_count);

    Tokenizer tokenizer;
    if (tokenizer_load(&tokenizer, tokenizer_path, config->vocab_size) != 0) return 1;
    if (run_compare) {
        int compare_status = compare_q8(model_path, &tokenizer, prompt, steps);
        ops_shutdown();
        tokenizer_free(&tokenizer);
        model_free(&model);
        return compare_status;
    }

    int prompt_tokens[512];
    int prompt_length = tokenizer_encode(&tokenizer, prompt, prompt_tokens, 512);

    if (rng_state == 0) rng_state = 1;  // a seed of 0 would get stuck at 0 forever
    int current_token = prompt_tokens[0], next_token;
    double start_ms = 0;

    for (int position = 0; position < steps; position++) {
        float *logits = model_forward(&model, current_token, position);
        if (position < prompt_length - 1) next_token = prompt_tokens[position + 1];  // still feeding the prompt
        else next_token = sample(logits, config->vocab_size, temperature, &rng_state);
        if (next_token == BOS_TOKEN) break;  // BOS ends the story
        printf("%s", tokenizer_decode(&tokenizer, current_token, next_token));
        fflush(stdout);
        current_token = next_token;
        if (position == 0) start_ms = prof_now_ms();  // skip the warm-up step
    }

    if (steps > 1) printf("\n\n%.1f tok/s\n", (steps - 1) / ((prof_now_ms() - start_ms) / 1000.0));

    ops_shutdown();
    tokenizer_free(&tokenizer);
    model_free(&model);
    return 0;
}
