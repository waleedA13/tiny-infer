#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

// Header of the checkpoint file (same layout as llama2.c's TinyStories .bin files).
typedef struct {
    int dim;        // embedding size
    int hidden_dim; // feed-forward hidden size
    int n_layers;
    int n_heads;
    int n_kv_heads; // < n_heads means grouped-query attention
    int vocab_size; // negative in the file means the classifier weights are separate
    int seq_len;    // max context length
} Config;

// Pointers into the mmap'd checkpoint. Nothing here is malloc'd.
typedef struct {
    float *token_embedding; // (vocab_size, dim)
    float *rms_att;         // (n_layers, dim)
    float *wq;              // (n_layers, dim, n_heads * head_size)
    float *wk;              // (n_layers, dim, n_kv_heads * head_size)
    float *wv;              // (n_layers, dim, n_kv_heads * head_size)
    float *wo;              // (n_layers, n_heads * head_size, dim)
    float *rms_ffn;         // (n_layers, dim)
    float *w1;              // (n_layers, hidden_dim, dim)
    float *w2;              // (n_layers, dim, hidden_dim)
    float *w3;              // (n_layers, hidden_dim, dim)
    float *rms_final;       // (dim)
    float *wcls;            // (vocab_size, dim), may alias token_embedding
} Weights;

// Activations reused every step. These are malloc'd once.
typedef struct {
    float *x, *xb, *xb2;     // (dim)
    float *hb, *hb2;         // (hidden_dim)
    float *q;                // (dim)
    float *att;              // (n_heads, seq_len)
    float *logits;           // (vocab_size)
    float *key_cache;        // (n_layers, seq_len, kv_dim)
    float *value_cache;      // (n_layers, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    Weights w;
    RunState s;
    int fd;
    void *mapped;
    size_t file_size;
} Model;

int  model_load(Model *m, const char *path);
void model_free(Model *m);

// Runs one token through the network at position pos and returns logits (vocab_size).
float *model_forward(Model *m, int token, int pos);

#endif
