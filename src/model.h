#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>

// Header of the checkpoint file (same layout as llama2.c's TinyStories .bin files).
// The order of these fields has to match the file, so don't reorder them.
typedef struct {
    int embed_dim;    // size of each token's vector
    int hidden_dim;   // feed-forward hidden size
    int num_layers;
    int num_heads;
    int num_kv_heads; // fewer than num_heads means several query heads share one key/value head
    int vocab_size;   // negative in the file means the classifier weights are stored separately
    int max_seq_len;  // longest context the model supports
} Config;

// Pointers into the mmap'd checkpoint. Nothing here is malloc'd.
typedef struct {
    float *token_embedding;       // (vocab_size, embed_dim)
    float *attention_norm;        // (num_layers, embed_dim)
    float *query_weights;         // (num_layers, embed_dim, num_heads * head_size)
    float *key_weights;           // (num_layers, embed_dim, num_kv_heads * head_size)
    float *value_weights;         // (num_layers, embed_dim, num_kv_heads * head_size)
    float *attention_out_weights; // (num_layers, num_heads * head_size, embed_dim)
    float *ffn_norm;              // (num_layers, embed_dim)
    float *ffn_gate_weights;      // (num_layers, hidden_dim, embed_dim), "w1"
    float *ffn_down_weights;      // (num_layers, embed_dim, hidden_dim), "w2"
    float *ffn_up_weights;        // (num_layers, hidden_dim, embed_dim), "w3"
    float *final_norm;            // (embed_dim)
    float *classifier_weights;    // (vocab_size, embed_dim), may point at token_embedding
} Weights;

// Scratch space reused every step. These are malloc'd once.
typedef struct {
    float *residual;          // (embed_dim) the running hidden state for this token
    float *normed;            // (embed_dim) residual after rmsnorm
    float *attention_output;  // (embed_dim) all heads' outputs side by side
    float *layer_output;      // (embed_dim) what gets added back into residual
    float *ffn_gate;          // (hidden_dim)
    float *ffn_up;            // (hidden_dim)
    float *query;             // (embed_dim)
    float *attention_scores;  // (num_heads, max_seq_len)
    float *logits;            // (vocab_size)
    float *key_cache;         // (num_layers, max_seq_len, kv_dim)
    float *value_cache;       // (num_layers, max_seq_len, kv_dim)
} RunState;

// An int8 copy of all the weights, made by model_quantize(). The values sit in the
// same order as the floats in the file. So if a matrix starts at float offset X,
// its int8 values start at X too, and its scales start at X / Q8_GROUP. This means
// the forward pass doesn't have to care which version it's using.
typedef struct {
    const float *float_base;   // start of the float weights (w.token_embedding)
    long value_count;          // number of floats from float_base to the end of the weights
    int8_t *values;            // (value_count)
    float *scales;             // (value_count / Q8_GROUP)
    int8_t *input_values;      // scratch for the int8 input of each matmul, max(dim, hidden_dim)
    float *input_scales;       // (max(dim, hidden_dim) / Q8_GROUP)
} QuantizedWeights;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
    QuantizedWeights quantized;  // quantized.values == NULL means fp32
    int file_descriptor;
    void *mapped_file;
    size_t file_size;
} Model;

int  model_load(Model *model, const char *path);
void model_free(Model *model);

// Turns every weight into int8 (see QuantizedWeights). After this, the matmuls in
// model_forward use matmul_q8. Returns -1 if the sizes aren't multiples of
// Q8_GROUP or we run out of memory. The model just stays fp32 in that case.
int  model_quantize(Model *model);

// Runs one token through the network at the given position and returns logits (vocab_size).
float *model_forward(Model *model, int token, int position);

#endif
