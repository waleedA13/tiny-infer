#include "model.h"
#include "ops.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Returns a pointer just past the last weight.
static float *map_weights(Weights *weights, const Config *config, float *cursor, int shares_classifier) {
    int head_size = config->embed_dim / config->num_heads;
    long num_layers = config->num_layers;
    long embed_dim = config->embed_dim;
    weights->token_embedding = cursor;       cursor += (long)config->vocab_size * embed_dim;
    weights->attention_norm = cursor;        cursor += num_layers * embed_dim;
    weights->query_weights = cursor;         cursor += num_layers * embed_dim * (config->num_heads * head_size);
    weights->key_weights = cursor;           cursor += num_layers * embed_dim * (config->num_kv_heads * head_size);
    weights->value_weights = cursor;         cursor += num_layers * embed_dim * (config->num_kv_heads * head_size);
    weights->attention_out_weights = cursor; cursor += num_layers * (config->num_heads * head_size) * embed_dim;
    weights->ffn_norm = cursor;              cursor += num_layers * embed_dim;
    weights->ffn_gate_weights = cursor;      cursor += num_layers * embed_dim * config->hidden_dim;
    weights->ffn_down_weights = cursor;      cursor += num_layers * config->hidden_dim * embed_dim;
    weights->ffn_up_weights = cursor;        cursor += num_layers * embed_dim * config->hidden_dim;
    weights->final_norm = cursor;            cursor += embed_dim;
    cursor += config->max_seq_len * head_size / 2;  // legacy freq_cis_real, unused
    cursor += config->max_seq_len * head_size / 2;  // legacy freq_cis_imag, unused
    weights->classifier_weights = shares_classifier ? weights->token_embedding : cursor;
    if (!shares_classifier) cursor += (long)config->vocab_size * embed_dim;
    return cursor;
}

static int alloc_state(RunState *state, const Config *config) {
    int kv_dim = config->embed_dim * config->num_kv_heads / config->num_heads;
    size_t cache_size = (size_t)config->num_layers * config->max_seq_len * kv_dim;
    state->residual = calloc(config->embed_dim, sizeof(float));
    state->normed = calloc(config->embed_dim, sizeof(float));
    state->attention_output = calloc(config->embed_dim, sizeof(float));
    state->layer_output = calloc(config->embed_dim, sizeof(float));
    state->ffn_gate = calloc(config->hidden_dim, sizeof(float));
    state->ffn_up = calloc(config->hidden_dim, sizeof(float));
    state->query = calloc(config->embed_dim, sizeof(float));
    state->attention_scores = calloc((size_t)config->num_heads * config->max_seq_len, sizeof(float));
    state->logits = calloc(config->vocab_size, sizeof(float));
    state->key_cache = calloc(cache_size, sizeof(float));
    state->value_cache = calloc(cache_size, sizeof(float));
    return (state->residual && state->normed && state->attention_output && state->layer_output &&
            state->ffn_gate && state->ffn_up && state->query && state->attention_scores &&
            state->logits && state->key_cache && state->value_cache) ? 0 : -1;
}

int model_load(Model *model, const char *path) {
    memset(model, 0, sizeof(*model));
    model->file_descriptor = open(path, O_RDONLY);
    if (model->file_descriptor < 0) { perror(path); return -1; }

    struct stat file_info;
    if (fstat(model->file_descriptor, &file_info) != 0) { perror("fstat"); return -1; }
    model->file_size = file_info.st_size;

    model->mapped_file = mmap(NULL, model->file_size, PROT_READ, MAP_PRIVATE, model->file_descriptor, 0);
    if (model->mapped_file == MAP_FAILED) { perror("mmap"); return -1; }

    memcpy(&model->config, model->mapped_file, sizeof(Config));
    int shares_classifier = model->config.vocab_size > 0;
    model->config.vocab_size = abs(model->config.vocab_size);

    float *weight_data = (float *)((char *)model->mapped_file + sizeof(Config));
    float *weights_end = map_weights(&model->weights, &model->config, weight_data, shares_classifier);
    if ((char *)weights_end > (char *)model->mapped_file + model->file_size) {
        fprintf(stderr, "%s: file is smaller than its config says\n", path);
        return -1;
    }
    model->quantized.float_base = weight_data;
    model->quantized.value_count = weights_end - weight_data;
    return alloc_state(&model->state, &model->config);
}

int model_quantize(Model *model) {
    Config *config = &model->config;
    QuantizedWeights *quantized = &model->quantized;
    int kv_dim = config->embed_dim * config->num_kv_heads / config->num_heads;
    if (config->embed_dim % Q8_GROUP || config->hidden_dim % Q8_GROUP || kv_dim % Q8_GROUP ||
        quantized->value_count % Q8_GROUP) {
        fprintf(stderr, "int8: embed_dim, hidden_dim and kv_dim must be multiples of %d\n", Q8_GROUP);
        return -1;
    }
    int largest_input_dim = config->embed_dim > config->hidden_dim ? config->embed_dim : config->hidden_dim;
    quantized->values = malloc(quantized->value_count);
    quantized->scales = malloc(quantized->value_count / Q8_GROUP * sizeof(float));
    quantized->input_values = malloc(largest_input_dim);
    quantized->input_scales = malloc(largest_input_dim / Q8_GROUP * sizeof(float));
    if (!quantized->values || !quantized->scales || !quantized->input_values || !quantized->input_scales) {
        free(quantized->values); free(quantized->scales);
        free(quantized->input_values); free(quantized->input_scales);
        quantized->values = NULL; quantized->scales = NULL;
        quantized->input_values = NULL; quantized->input_scales = NULL;
        return -1;
    }
    quantize_q8(quantized->values, quantized->scales, quantized->float_base, quantized->value_count);
    return 0;
}

// Every weight matmul in the forward pass goes through here. float_weights is the
// float matrix from the file. If the model is quantized, we use its int8 copy instead.
static void linear(Model *model, float *output, const float *input, const float *float_weights,
                   int in_size, int out_size) {
    QuantizedWeights *quantized = &model->quantized;
    if (!quantized->values) { matmul(output, input, float_weights, in_size, out_size); return; }
    long weight_offset = float_weights - quantized->float_base;
    quantize_q8(quantized->input_values, quantized->input_scales, input, in_size);
    matmul_q8(output, quantized->input_values, quantized->input_scales,
              quantized->values + weight_offset, quantized->scales + weight_offset / Q8_GROUP,
              in_size, out_size);
}

void model_free(Model *model) {
    RunState *state = &model->state;
    free(state->residual); free(state->normed); free(state->attention_output);
    free(state->layer_output); free(state->ffn_gate); free(state->ffn_up);
    free(state->query); free(state->attention_scores); free(state->logits);
    free(state->key_cache); free(state->value_cache);
    free(model->quantized.values); free(model->quantized.scales);
    free(model->quantized.input_values); free(model->quantized.input_scales);
    if (model->mapped_file && model->mapped_file != MAP_FAILED) munmap(model->mapped_file, model->file_size);
    if (model->file_descriptor > 0) close(model->file_descriptor);
}

// RoPE: spin each pair of numbers in a head by an angle.
// The angle depends on the token's position and where the pair sits in the head.
static void rope(float *vector, int size, int head_size, int position) {
    for (int i = 0; i < size; i += 2) {
        int index_in_head = i % head_size;
        float frequency = 1.0f / powf(10000.0f, (float)index_in_head / head_size);
        float angle = position * frequency;
        float cos_angle = cosf(angle), sin_angle = sinf(angle);
        float first = vector[i], second = vector[i + 1];
        vector[i]     = first * cos_angle - second * sin_angle;
        vector[i + 1] = first * sin_angle + second * cos_angle;
    }
}

float *model_forward(Model *model, int token, int position) {
    Config *config = &model->config;
    Weights *weights = &model->weights;
    RunState *state = &model->state;
    int embed_dim = config->embed_dim, hidden_dim = config->hidden_dim;
    int head_size = embed_dim / config->num_heads;
    int kv_dim = head_size * config->num_kv_heads;
    int heads_per_kv_head = config->num_heads / config->num_kv_heads;
    float score_scale = 1.0f / sqrtf((float)head_size);

    // 1. Look up the token embedding.
    memcpy(state->residual, weights->token_embedding + (long)token * embed_dim, embed_dim * sizeof(float));

    for (int layer = 0; layer < config->num_layers; layer++) {
        // Weight matmuls go through linear() so they pick up int8.
        // Attention. This token's key and value get saved in the cache.
        long layer_cache_offset = (long)layer * config->max_seq_len * kv_dim;
        float *layer_key_cache = state->key_cache + layer_cache_offset;
        float *layer_value_cache = state->value_cache + layer_cache_offset;
        float *new_key = layer_key_cache + (long)position * kv_dim;
        float *new_value = layer_value_cache + (long)position * kv_dim;

        rmsnorm(state->normed, state->residual, weights->attention_norm + (long)layer * embed_dim, embed_dim);
        linear(model, state->query, state->normed, weights->query_weights + (long)layer * embed_dim * embed_dim, embed_dim, embed_dim);
        linear(model, new_key, state->normed, weights->key_weights + (long)layer * embed_dim * kv_dim, embed_dim, kv_dim);
        linear(model, new_value, state->normed, weights->value_weights + (long)layer * embed_dim * kv_dim, embed_dim, kv_dim);
        rope(state->query, embed_dim, head_size, position);
        rope(new_key, kv_dim, head_size, position);

        for (int head = 0; head < config->num_heads; head++) {
            const float *head_query = state->query + head * head_size;
            float *head_scores = state->attention_scores + (long)head * config->max_seq_len;
            int kv_head_offset = (head / heads_per_kv_head) * head_size;

            // Compare this query with every key we've seen so far.
            for (int past = 0; past <= position; past++) {
                const float *past_key = layer_key_cache + (long)past * kv_dim + kv_head_offset;
                float dot_product = 0.0f;
                for (int i = 0; i < head_size; i++) dot_product += head_query[i] * past_key[i];
                head_scores[past] = dot_product * score_scale;
            }
            softmax(head_scores, position + 1);

            // Mix the saved values using those scores.
            float *head_output = state->attention_output + head * head_size;
            memset(head_output, 0, head_size * sizeof(float));
            for (int past = 0; past <= position; past++) {
                const float *past_value = layer_value_cache + (long)past * kv_dim + kv_head_offset;
                for (int i = 0; i < head_size; i++) head_output[i] += head_scores[past] * past_value[i];
            }
        }

        linear(model, state->layer_output, state->attention_output,
               weights->attention_out_weights + (long)layer * embed_dim * embed_dim, embed_dim, embed_dim);
        for (int i = 0; i < embed_dim; i++) state->residual[i] += state->layer_output[i];

        // Feed-forward: down(silu(gate(x)) * up(x)).
        rmsnorm(state->normed, state->residual, weights->ffn_norm + (long)layer * embed_dim, embed_dim);
        linear(model, state->ffn_gate, state->normed, weights->ffn_gate_weights + (long)layer * embed_dim * hidden_dim, embed_dim, hidden_dim);
        linear(model, state->ffn_up, state->normed, weights->ffn_up_weights + (long)layer * embed_dim * hidden_dim, embed_dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float gate = state->ffn_gate[i];
            state->ffn_gate[i] = gate / (1.0f + expf(-gate)) * state->ffn_up[i];
        }
        linear(model, state->layer_output, state->ffn_gate,
               weights->ffn_down_weights + (long)layer * embed_dim * hidden_dim, hidden_dim, embed_dim);
        for (int i = 0; i < embed_dim; i++) state->residual[i] += state->layer_output[i];
    }

    // Final norm and classifier into logits.
    rmsnorm(state->residual, state->residual, weights->final_norm, embed_dim);
    linear(model, state->logits, state->residual, weights->classifier_weights, embed_dim, config->vocab_size);
    return state->logits;
}
