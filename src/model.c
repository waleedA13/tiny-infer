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

static void map_weights(Weights *w, const Config *c, float *p, int shared_cls) {
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
    map_weights(&m->w, &m->config, weights, shared_cls);
    return alloc_state(&m->s, &m->config);
}

void model_free(Model *m) {
    RunState *s = &m->s;
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits); free(s->key_cache); free(s->value_cache);
    if (m->mapped && m->mapped != MAP_FAILED) munmap(m->mapped, m->file_size);
    if (m->fd > 0) close(m->fd);
}

// RoPE: spin each pair of numbers in a head by an angle.
// The angle depends on the token's position and where the pair sits in the head.
static void rope(float *v, int n, int head_size, int pos) {
    for (int i = 0; i < n; i += 2) {
        int j = i % head_size;
        float freq = 1.0f / powf(10000.0f, (float)j / head_size);
        float angle = pos * freq;
        float cr = cosf(angle), ci = sinf(angle);
        float a = v[i], b = v[i + 1];
        v[i]     = a * cr - b * ci;
        v[i + 1] = a * ci + b * cr;
    }
}

float *model_forward(Model *m, int token, int pos) {
    Config *c = &m->config;
    Weights *w = &m->w;
    RunState *s = &m->s;
    int dim = c->dim, hidden = c->hidden_dim;
    int head_size = dim / c->n_heads;
    int kv_dim = head_size * c->n_kv_heads;
    int kv_mul = c->n_heads / c->n_kv_heads;  // query heads sharing one kv head
    float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);

    // 1. Look up the token embedding.
    memcpy(s->x, w->token_embedding + (long)token * dim, dim * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        // Attention. This token's k and v get saved in the cache.
        long cache_off = (long)l * c->seq_len * kv_dim;
        float *k = s->key_cache + cache_off + (long)pos * kv_dim;
        float *v = s->value_cache + cache_off + (long)pos * kv_dim;

        rmsnorm(s->xb, s->x, w->rms_att + (long)l * dim, dim);
        matmul(s->q, s->xb, w->wq + (long)l * dim * dim, dim, dim);
        matmul(k, s->xb, w->wk + (long)l * dim * kv_dim, dim, kv_dim);
        matmul(v, s->xb, w->wv + (long)l * dim * kv_dim, dim, kv_dim);
        rope(s->q, dim, head_size, pos);
        rope(k, kv_dim, head_size, pos);

        for (int h = 0; h < c->n_heads; h++) {
            const float *q = s->q + h * head_size;
            float *att = s->att + (long)h * c->seq_len;
            int kv_off = (h / kv_mul) * head_size;

            // Compare this query with every key we've seen so far.
            for (int t = 0; t <= pos; t++) {
                const float *kt = s->key_cache + cache_off + (long)t * kv_dim + kv_off;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * kt[i];
                att[t] = score * inv_sqrt_hs;
            }
            softmax(att, pos + 1);

            // Mix the saved values using those scores.
            float *out = s->xb + h * head_size;
            memset(out, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *vt = s->value_cache + cache_off + (long)t * kv_dim + kv_off;
                for (int i = 0; i < head_size; i++) out[i] += att[t] * vt[i];
            }
        }

        matmul(s->xb2, s->xb, w->wo + (long)l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        // Feed-forward: w2(silu(w1 x) * w3 x).
        rmsnorm(s->xb, s->x, w->rms_ffn + (long)l * dim, dim);
        matmul(s->hb, s->xb, w->w1 + (long)l * dim * hidden, dim, hidden);
        matmul(s->hb2, s->xb, w->w3 + (long)l * dim * hidden, dim, hidden);
        for (int i = 0; i < hidden; i++) {
            float g = s->hb[i];
            s->hb[i] = g / (1.0f + expf(-g)) * s->hb2[i];
        }
        matmul(s->xb, s->hb, w->w2 + (long)l * dim * hidden, hidden, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    // Final norm and classifier into logits.
    rmsnorm(s->x, s->x, w->rms_final, c->dim);
    matmul(s->logits, s->x, w->wcls, c->dim, c->vocab_size);
    return s->logits;
}
