#define _POSIX_C_SOURCE 200809L
#include "compare.h"
#include "model.h"
#include "ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int argmax(const float *values, int count) {
    int best_index = 0;
    for (int i = 1; i < count; i++) if (values[i] > values[best_index]) best_index = i;
    return best_index;
}

// KL(p || q) with p = softmax(reference_logits) and q = softmax(test_logits).
// Uses doubles so the check itself doesn't add error.
static double kl_from_logits(const float *reference_logits, const float *test_logits, int vocab_size) {
    double reference_max = reference_logits[argmax(reference_logits, vocab_size)];
    double test_max = test_logits[argmax(test_logits, vocab_size)];
    double reference_sum = 0, test_sum = 0;
    for (int i = 0; i < vocab_size; i++) {
        reference_sum += exp(reference_logits[i] - reference_max);
        test_sum += exp(test_logits[i] - test_max);
    }
    double kl = 0;
    for (int i = 0; i < vocab_size; i++) {
        double reference_log_prob = reference_logits[i] - reference_max - log(reference_sum);
        double test_log_prob = test_logits[i] - test_max - log(test_sum);
        kl += exp(reference_log_prob) * (reference_log_prob - test_log_prob);
    }
    return kl;
}

// Greedy generation from the prompt. Puts the tokens after BOS in out_tokens and
// returns how many there are. *tokens_per_second gets the speed.
static int generate_greedy(Model *m, const int *prompt_tokens, int prompt_length, int steps,
                           int *out_tokens, double *tokens_per_second) {
    double timed_seconds = 0;
    int timed_steps = 0, token_count = 0, token = prompt_tokens[0];
    for (int pos = 0; pos < steps; pos++) {
        double step_start = now_seconds();
        float *logits = model_forward(m, token, pos);
        if (pos > 0) { timed_seconds += now_seconds() - step_start; timed_steps++; }  // skip warm-up step, like main
        int next = pos < prompt_length - 1 ? prompt_tokens[pos + 1] : argmax(logits, m->config.vocab_size);
        if (next == 1) break;
        out_tokens[token_count++] = next;
        token = next;
    }
    *tokens_per_second = timed_steps > 0 ? timed_steps / timed_seconds : 0;
    return token_count;
}

static void print_story(const Tokenizer *t, int first_token, const int *tokens, int token_count) {
    int previous_token = first_token;
    for (int i = 0; i < token_count; i++) {
        printf("%s", tokenizer_decode(t, previous_token, tokens[i]));
        previous_token = tokens[i];
    }
    printf("\n");
}

int compare_q8(const char *model_path, const Tokenizer *t, const char *prompt, int steps) {
    Model fp32_model, int8_model;
    if (model_load(&fp32_model, model_path) != 0 || model_load(&int8_model, model_path) != 0) return 1;
    if (model_quantize(&int8_model) != 0) return 1;
    int vocab_size = fp32_model.config.vocab_size;
    if (steps > fp32_model.config.max_seq_len) steps = fp32_model.config.max_seq_len;

    int prompt_tokens[512];
    int prompt_length = tokenizer_encode(t, prompt, prompt_tokens, 512);

    // 1. Feed both models the exact same tokens (the fp32 greedy story). This way
    //    we see the error at each step without one bad pick throwing off the rest.
    double kl_total = 0, kl_worst = 0, largest_logit_gap = 0;
    int same_top_token = 0, positions_checked = 0, token = prompt_tokens[0];
    for (int pos = 0; pos < steps; pos++) {
        float *fp32_logits = model_forward(&fp32_model, token, pos);
        float *int8_logits = model_forward(&int8_model, token, pos);
        for (int i = 0; i < vocab_size; i++) {
            double logit_gap = fabs(fp32_logits[i] - int8_logits[i]);
            if (logit_gap > largest_logit_gap) largest_logit_gap = logit_gap;
        }
        double kl = kl_from_logits(fp32_logits, int8_logits, vocab_size);
        kl_total += kl;
        if (kl > kl_worst) kl_worst = kl;
        int fp32_top_token = argmax(fp32_logits, vocab_size);
        same_top_token += fp32_top_token == argmax(int8_logits, vocab_size);
        positions_checked++;
        int next = pos < prompt_length - 1 ? prompt_tokens[pos + 1] : fp32_top_token;
        if (next == 1) break;
        token = next;
    }

    // 2. Now let each model write its own greedy story.
    int *fp32_story = malloc(steps * sizeof(int)), *int8_story = malloc(steps * sizeof(int));
    double fp32_speed, int8_speed;
    int fp32_length = generate_greedy(&fp32_model, prompt_tokens, prompt_length, steps, fp32_story, &fp32_speed);
    int int8_length = generate_greedy(&int8_model, prompt_tokens, prompt_length, steps, int8_story, &int8_speed);
    int shorter_length = fp32_length < int8_length ? fp32_length : int8_length;
    int matching_prefix = 0;
    while (matching_prefix < shorter_length && fp32_story[matching_prefix] == int8_story[matching_prefix])
        matching_prefix++;

    long weight_count = fp32_model.quantized.value_count;
    printf("int8 vs fp32, group size %d, %d positions\n", Q8_GROUP, positions_checked);
    printf("  weights        %.1f MB -> %.1f MB\n", weight_count * 4 / 1e6,
           (weight_count + weight_count / Q8_GROUP * 4) / 1e6);
    printf("  max |dlogit|   %.4f\n", largest_logit_gap);
    printf("  mean KL        %.6f (max %.6f)\n", kl_total / positions_checked, kl_worst);
    printf("  top-1 agree    %d / %d (%.1f%%)\n", same_top_token, positions_checked,
           100.0 * same_top_token / positions_checked);
    printf("  greedy match   first %d of %d tokens identical\n", matching_prefix, shorter_length);
    printf("  speed          fp32 %.1f tok/s, int8 %.1f tok/s (%.2fx)\n",
           fp32_speed, int8_speed, int8_speed / fp32_speed);
    printf("\n--- fp32 ---\n");
    print_story(t, prompt_tokens[0], fp32_story, fp32_length);
    printf("\n--- int8 ---\n");
    print_story(t, prompt_tokens[0], int8_story, int8_length);

    free(fp32_story); free(int8_story);
    model_free(&fp32_model); model_free(&int8_model);
    return 0;
}
