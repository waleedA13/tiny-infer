#define _POSIX_C_SOURCE 200809L  // needed for clock_gettime with -std=c11

#include "compare.h"
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
    int failures = 0;

    float matrix[6] = {1, 2, 3, 4, 5, 6}, ones_vector[3] = {1, 1, 1}, matmul_result[2];
    matmul(matmul_result, ones_vector, matrix, 3, 2);
    if (matmul_result[0] != 6 || matmul_result[1] != 15) { printf("FAIL matmul\n"); failures++; }

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
    if (!round_trip_ok) { printf("FAIL quantize_q8 round trip\n"); failures++; }
    quantize_q8(int8_input, input_scales, float_input, TEST_IN_DIM);
    matmul(expected_out, float_input, float_weights, TEST_IN_DIM, TEST_OUT_DIM);
    matmul_q8(int8_out, int8_input, input_scales, int8_weights, weight_scales, TEST_IN_DIM, TEST_OUT_DIM);
    for (int i = 0; i < TEST_OUT_DIM; i++)
        if (fabsf(int8_out[i] - expected_out[i]) > 0.02f * (fabsf(expected_out[i]) + 1.0f)) {
            printf("FAIL matmul_q8 (row %d: %f vs %f)\n", i, int8_out[i], expected_out[i]); failures++; break;
        }

    // The threaded matmul should give exactly the same numbers as one thread.
    // Check the stories15M sizes, plus a couple that don't split evenly.
    int test_shapes[][2] = {{288, 288}, {288, 768}, {768, 288}, {288, 32000}, {7, 5}, {13, 3}};
    int shape_count = sizeof(test_shapes) / sizeof(test_shapes[0]);
    for (int shape_index = 0; shape_index < shape_count; shape_index++) {
        int input_size = test_shapes[shape_index][0], output_size = test_shapes[shape_index][1];
        float *test_weights = malloc((size_t)input_size * output_size * sizeof(float));
        float *test_input = malloc(input_size * sizeof(float));
        float *single_thread_result = malloc(output_size * sizeof(float));
        float *multi_thread_result = malloc(output_size * sizeof(float));

        // Fill with fixed pseudo-random values between -0.5 and 0.5.
        unsigned long long random_state = 42;
        long weight_count = (long)input_size * output_size;
        for (long i = 0; i < weight_count; i++) {
            random_state = random_state * 6364136223846793005ULL + 1;
            test_weights[i] = (float)(random_state >> 40) / 16777216.0f - 0.5f;
        }
        for (int i = 0; i < input_size; i++) {
            random_state = random_state * 6364136223846793005ULL + 1;
            test_input[i] = (float)(random_state >> 40) / 16777216.0f - 0.5f;
        }

        matmul(single_thread_result, test_input, test_weights, input_size, output_size);
        for (int thread_count = 2; thread_count <= 8; thread_count *= 2) {
            ops_init(thread_count);
            matmul(multi_thread_result, test_input, test_weights, input_size, output_size);
            ops_shutdown();
            if (memcmp(single_thread_result, multi_thread_result, output_size * sizeof(float)) != 0) {
                printf("FAIL threaded matmul (input_size=%d output_size=%d threads=%d)\n",
                       input_size, output_size, thread_count);
                failures++;
            }
        }
        free(test_weights); free(test_input); free(single_thread_result); free(multi_thread_result);
    }

    float softmax_values[3] = {1, 2, 3};
    softmax(softmax_values, 3);
    float probability_sum = softmax_values[0] + softmax_values[1] + softmax_values[2];
    if (fabsf(probability_sum - 1.0f) > 1e-5f) { printf("FAIL softmax (sum = %f)\n", probability_sum); failures++; }

    float norm_input[2] = {3, 4}, norm_weight[2] = {1, 1}, norm_output[2];
    rmsnorm(norm_output, norm_input, norm_weight, 2);
    float expected = 3.0f / sqrtf(12.5f + 1e-5f);
    if (fabsf(norm_output[0] - expected) > 1e-4f) { printf("FAIL rmsnorm\n"); failures++; }

    printf(failures ? "%d test(s) failed\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}

static int compare_doubles(const void *left_ptr, const void *right_ptr) {
    double left = *(const double *)left_ptr, right = *(const double *)right_ptr;
    return (left > right) - (left < right);
}

// Benchmark mode. Same thing every time: start from BOS, always pick the top token,
// run a fixed number of steps. One warm-up run first, then report the middle of the
// timed runs. No tokenizer and no printing while the clock is running.
static int run_benchmark(const char *model_path, int num_steps, int num_runs, int thread_count) {
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
    ops_shutdown();
    model_free(&model);
    return 0;
}

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
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
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
    int token = prompt_tokens[0], next_token;
    double start_time = 0;

    for (int position = 0; position < steps; position++) {
        float *logits = model_forward(&model, token, position);
        if (position < prompt_length - 1) next_token = prompt_tokens[position + 1];  // still feeding the prompt
        else next_token = sample(logits, config->vocab_size, temperature, &rng_state);
        if (next_token == 1) break;                                               // BOS ends the story
        printf("%s", tokenizer_decode(&tokenizer, token, next_token));
        fflush(stdout);
        token = next_token;
        if (position == 0) start_time = prof_now_ms();                                 // skip warm-up step
    }

    if (steps > 1) printf("\n\n%.1f tok/s\n", (steps - 1) / ((prof_now_ms() - start_time) / 1000.0));

    ops_shutdown();
    tokenizer_free(&tokenizer);
    model_free(&model);
    return 0;
}
