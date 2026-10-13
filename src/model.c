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
static float *map_weights(Weights *w, const Config *c, float *p, int shared_cls) {
    int head_size = c->dim / c->n_heads;
    long L = c->n_layers;
    w->token_embedding = p; p += (long)c->vocab_size * c->dim;
    w->rms_att = p;         p += L * c->dim;
    w->wq = p;              p += L * c->dim * (c->n_heads * head_size);
    w->wk = p;              p += L * c->dim * (c->n_kv_heads * head_size);
    w->wv = p;              p += L * c->dim * (c->n_kv_heads * head_size);
    w->wo = p;              p += L * (c->n_heads * head_size) * c->dim;
    w->rms_ffn = p;         p += L * c->dim;
    w->w1 = p;              p += L * c->dim * c->hidden_dim;
    w->w2 = p;              p += L * c->hidden_dim * c->dim;
    w->w3 = p;              p += L * c->dim * c->hidden_dim;
    w->rms_final = p;       p += c->dim;
    p += c->seq_len * head_size / 2;  // legacy freq_cis_real, unused
    p += c->seq_len * head_size / 2;  // legacy freq_cis_imag, unused
    w->wcls = shared_cls ? w->token_embedding : p;
    if (!shared_cls) p += (long)c->vocab_size * c->dim;
    return p;
}

static int alloc_state(RunState *s, const Config *c) {
    int kv_dim = c->dim * c->n_kv_heads / c->n_heads;
    s->x = calloc(c->dim, sizeof(float));
    s->xb = calloc(c->dim, sizeof(float));
    s->xb2 = calloc(c->dim, sizeof(float));
    s->hb = calloc(c->hidden_dim, sizeof(float));
    s->hb2 = calloc(c->hidden_dim, sizeof(float));
    s->q = calloc(c->dim, sizeof(float));
    s->att = calloc((size_t)c->n_heads * c->seq_len, sizeof(float));
    s->logits = calloc(c->vocab_size, sizeof(float));
    s->key_cache = calloc((size_t)c->n_layers * c->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc((size_t)c->n_layers * c->seq_len * kv_dim, sizeof(float));
    return (s->x && s->xb && s->xb2 && s->hb && s->hb2 && s->q && s->att &&
            s->logits && s->key_cache && s->value_cache) ? 0 : -1;
}

int model_load(Model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) { perror(path); return -1; }

    struct stat st;
    if (fstat(m->fd, &st) != 0) { perror("fstat"); return -1; }
    m->file_size = st.st_size;

    m->mapped = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->mapped == MAP_FAILED) { perror("mmap"); return -1; }

    memcpy(&m->config, m->mapped, sizeof(Config));
    int shared_cls = m->config.vocab_size > 0;
    m->config.vocab_size = abs(m->config.vocab_size);

    float *weights = (float *)((char *)m->mapped + sizeof(Config));
    float *weights_end = map_weights(&m->w, &m->config, weights, shared_cls);
    if ((char *)weights_end > (char *)m->mapped + m->file_size) {
        fprintf(stderr, "%s: file is smaller than its config says\n", path);
        return -1;
    }
    m->quantized.float_base = weights;
    m->quantized.value_count = weights_end - weights;
    return alloc_state(&m->s, &m->config);
}

int model_quantize(Model *m) {
    Config *c = &m->config;
    QuantizedWeights *quantized = &m->quantized;
    int kv_dim = c->dim * c->n_kv_heads / c->n_heads;
    if (c->dim % Q8_GROUP || c->hidden_dim % Q8_GROUP || kv_dim % Q8_GROUP ||
        quantized->value_count % Q8_GROUP) {
        fprintf(stderr, "int8: dim, hidden_dim and kv_dim must be multiples of %d\n", Q8_GROUP);
        return -1;
    }
    int largest_input_dim = c->dim > c->hidden_dim ? c->dim : c->hidden_dim;
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

// Every weight matmul in the forward pass goes through here. w is the float
// matrix from the file. If the model is quantized, we use its int8 copy instead.
static void linear(Model *m, float *out, const float *input, const float *float_weights,
                   int in_dim, int out_dim) {
    QuantizedWeights *quantized = &m->quantized;
    if (!quantized->values) { matmul(out, input, float_weights, in_dim, out_dim); return; }
    long weight_offset = float_weights - quantized->float_base;
    quantize_q8(quantized->input_values, quantized->input_scales, input, in_dim);
    matmul_q8(out, quantized->input_values, quantized->input_scales,
              quantized->values + weight_offset, quantized->scales + weight_offset / Q8_GROUP,
              in_dim, out_dim);
}

void model_free(Model *m) {
    RunState *s = &m->s;
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits); free(s->key_cache); free(s->value_cache);
    free(m->quantized.values); free(m->quantized.scales);
    free(m->quantized.input_values); free(m->quantized.input_scales);
    if (m->mapped && m->mapped != MAP_FAILED) munmap(m->mapped, m->file_size);
    if (m->fd > 0) close(m->fd);
}

float *model_forward(Model *m, int token, int pos) {
    Config *c = &m->config;
    Weights *w = &m->w;
    RunState *s = &m->s;
    (void)pos;

    // 1. Look up the token embedding.
    memcpy(s->x, w->token_embedding + (long)token * c->dim, c->dim * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        // Use linear() rather than matmul() for the weight matmuls so they pick up int8.
        // TODO(attention): rmsnorm -> q,k,v matmuls -> RoPE -> write k,v into the cache
        //   at pos -> scores against every cached key -> softmax -> weighted sum of values
        //   -> wo matmul -> residual add into s->x.

        // TODO(ffn): rmsnorm -> w1 and w3 matmuls -> SwiGLU (silu(w1 x) * (w3 x))
        //   -> w2 matmul -> residual add into s->x.
    }

    // Final norm and classifier into logits.
    rmsnorm(s->x, s->x, w->rms_final, c->dim);
    linear(m, s->logits, s->x, w->wcls, c->dim, c->vocab_size);
    return s->logits;
}
