#include "model.h"
#include "ops.h"
#include "prof.h"

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
    p += c->seq_len * head_size;  // legacy freq_cis_real, unused
    p += c->seq_len * head_size;  // legacy freq_cis_imag, unused
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

float *model_forward(Model *m, int token, int pos) {
    Config *c = &m->config;
    Weights *w = &m->w;
    RunState *s = &m->s;
    (void)pos;

    // 1. Look up the token embedding.
    memcpy(s->x, w->token_embedding + (long)token * c->dim, c->dim * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        // TODO(attention): rmsnorm -> q,k,v matmuls -> RoPE -> write k,v into the cache at pos.

        // The attention core is timed on its own for --bench (matmuls are timed inside matmul()).
        PROF_START(t_att);
        // TODO(attention): scores against every cached key -> softmax -> weighted sum of values.
        PROF_STOP(PROF_ATTENTION, t_att);

        // TODO(attention): wo matmul -> residual add into s->x.

        // TODO(ffn): rmsnorm -> w1 and w3 matmuls -> SwiGLU (silu(w1 x) * (w3 x))
        //   -> w2 matmul -> residual add into s->x.
    }

    // Final norm and classifier into logits.
    rmsnorm(s->x, s->x, w->rms_final, c->dim);
    matmul(s->logits, s->x, w->wcls, c->dim, c->vocab_size);
    return s->logits;
}
