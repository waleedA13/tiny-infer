#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>

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

// int8 copy of the whole weight blob, made by model_quantize(). It keeps the same
// element order as the mmap'd floats, so the float pointer for any matrix (say
// w.wq + l * dim * dim) maps to the int8 data at the same offset, and to its
// scales at offset / Q8_GROUP. That way the forward pass never needs to know
// which format it is running.
typedef struct {
    const float *base;  // start of the float weights (w.token_embedding)
    long n;             // number of floats from base to the end of the weights
    int8_t *q;          // (n)
    float *scale;       // (n / Q8_GROUP)
    int8_t *xq;         // activation scratch, max(dim, hidden_dim)
    float *xs;
} QWeights;

typedef struct {
    Config config;
    Weights w;
    RunState s;
    QWeights q;     // q.q == NULL means fp32
    int fd;
    void *mapped;
    size_t file_size;
} Model;

int  model_load(Model *m, const char *path);
void model_free(Model *m);

// Quantizes every weight to group-wise int8 (see QWeights). After this, all
// matmuls in model_forward run through matmul_q8. Returns -1 if the shapes are
// not multiples of Q8_GROUP or memory runs out; the model stays fp32 then.
int  model_quantize(Model *m);

// Runs one token through the network at position pos and returns logits (vocab_size).
float *model_forward(Model *m, int token, int pos);

#endif
