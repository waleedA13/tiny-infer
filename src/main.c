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
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1e6;
}

// Small checks for the math kernels. Add one every time you optimize a kernel,
// and compare the fast version against the naive one.
static int selftest(void) {
    int failures = 0;

    float matrix[6] = {1, 2, 3, 4, 5, 6}, ones_vector[3] = {1, 1, 1}, matmul_result[2];
    matmul(matmul_result, ones_vector, matrix, 3, 2);
    if (matmul_result[0] != 6 || matmul_result[1] != 15) { printf("FAIL matmul\n"); failures++; }

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

static void usage(void) {
    fprintf(stderr,
        "usage: tiny-infer <model.bin> [-z tokenizer.bin] [-n steps] [-t temperature] [-i prompt]\n"
        "       tiny-infer --selftest\n");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 2) { usage(); return 1; }

    const char *model_path = argv[1];
    const char *tokenizer_path = "models/tokenizer.bin";
    const char *prompt = "";
    int steps = 256;
    float temperature = 0.8f;
    for (int arg_index = 2; arg_index + 1 < argc; arg_index += 2) {
        const char *flag = argv[arg_index], *value = argv[arg_index + 1];
        if (!strcmp(flag, "-z")) tokenizer_path = value;
        else if (!strcmp(flag, "-n")) steps = atoi(value);
        else if (!strcmp(flag, "-t")) temperature = atof(value);
        else if (!strcmp(flag, "-i")) prompt = value;
        else { usage(); return 1; }
    }

    Model model;
    if (model_load(&model, model_path) != 0) return 1;
    Config *config = &model.config;
    printf("embed_dim=%d layers=%d heads=%d kv_heads=%d vocab=%d max_seq_len=%d\n",
           config->embed_dim, config->num_layers, config->num_heads, config->num_kv_heads,
           config->vocab_size, config->max_seq_len);
    if (steps > config->max_seq_len) steps = config->max_seq_len;

    Tokenizer tokenizer;
    if (tokenizer_load(&tokenizer, tokenizer_path, config->vocab_size) != 0) return 1;

    int prompt_tokens[512];
    int prompt_length = tokenizer_encode(&tokenizer, prompt, prompt_tokens, 512);

    unsigned long long rng_state = (unsigned long long)time(NULL);
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
        if (position == 0) start_time = now_ms();                                 // skip warm-up step
    }

    if (steps > 1) printf("\n\n%.1f tok/s\n", (steps - 1) / ((now_ms() - start_time) / 1000.0));

    tokenizer_free(&tokenizer);
    model_free(&model);
    return 0;
}
