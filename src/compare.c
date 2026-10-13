#define _POSIX_C_SOURCE 200809L
#include "compare.h"
#include "model.h"
#include "ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

// KL(p || q) with p = softmax(a) and q = softmax(b). Uses doubles so the
// check itself doesn't add error.
static double kl_logits(const float *a, const float *b, int n) {
    double ma = a[argmax(a, n)], mb = b[argmax(b, n)], za = 0, zb = 0;
    for (int i = 0; i < n; i++) { za += exp(a[i] - ma); zb += exp(b[i] - mb); }
    double kl = 0;
    for (int i = 0; i < n; i++) {
        double lpa = a[i] - ma - log(za), lpb = b[i] - mb - log(zb);
        kl += exp(lpa) * (lpa - lpb);
    }
    return kl;
}

// Greedy generation from the prompt. Puts the tokens after BOS in out[] and
// returns how many there are. *tok_s gets the speed.
static int generate(Model *m, const int *prompt, int n_prompt, int steps, int *out, double *tok_s) {
    double secs = 0;
    int timed = 0;
    int token = prompt[0], n = 0;
    for (int pos = 0; pos < steps; pos++) {
        double t0 = now_s();
        float *logits = model_forward(m, token, pos);
        if (pos > 0) { secs += now_s() - t0; timed++; }   // skip warm-up step, like main
        int next = pos < n_prompt - 1 ? prompt[pos + 1] : argmax(logits, m->config.vocab_size);
        if (next == 1) break;
        out[n++] = next;
        token = next;
    }
    *tok_s = timed > 0 ? timed / secs : 0;
    return n;
}

static void print_tokens(const Tokenizer *t, int first, const int *toks, int n) {
    int prev = first;
    for (int i = 0; i < n; i++) { printf("%s", tokenizer_decode(t, prev, toks[i])); prev = toks[i]; }
    printf("\n");
}

int compare_q8(const char *model_path, const Tokenizer *t, const char *prompt, int steps) {
    Model f, q;
    if (model_load(&f, model_path) != 0 || model_load(&q, model_path) != 0) return 1;
    if (model_quantize(&q) != 0) return 1;
    int vocab = f.config.vocab_size;
    if (steps > f.config.seq_len) steps = f.config.seq_len;

    int prompt_tokens[512];
    int n_prompt = tokenizer_encode(t, prompt, prompt_tokens, 512);

    // 1. Feed both models the exact same tokens (the fp32 greedy story). This way
    //    we see the error at each step without one bad pick throwing off the rest.
    double kl_sum = 0, kl_max = 0, dlogit_max = 0;
    int agree = 0, n = 0, token = prompt_tokens[0];
    for (int pos = 0; pos < steps; pos++) {
        float *a = model_forward(&f, token, pos);
        float *b = model_forward(&q, token, pos);
        for (int i = 0; i < vocab; i++) {
            double d = fabs(a[i] - b[i]);
            if (d > dlogit_max) dlogit_max = d;
        }
        double kl = kl_logits(a, b, vocab);
        kl_sum += kl;
        if (kl > kl_max) kl_max = kl;
        int top = argmax(a, vocab);
        agree += top == argmax(b, vocab);
        n++;
        int next = pos < n_prompt - 1 ? prompt_tokens[pos + 1] : top;
        if (next == 1) break;
        token = next;
    }

    // 2. Now let each model write its own greedy story.
    int *toks_f = malloc(steps * sizeof(int)), *toks_q = malloc(steps * sizeof(int));
    double speed_f, speed_q;
    int nf = generate(&f, prompt_tokens, n_prompt, steps, toks_f, &speed_f);
    int nq = generate(&q, prompt_tokens, n_prompt, steps, toks_q, &speed_q);
    int same = 0;
    while (same < nf && same < nq && toks_f[same] == toks_q[same]) same++;

    long nw = f.q.n;
    printf("int8 vs fp32, group size %d, %d positions\n", Q8_GROUP, n);
    printf("  weights        %.1f MB -> %.1f MB\n", nw * 4 / 1e6,
           (nw + nw / Q8_GROUP * 4) / 1e6);
    printf("  max |dlogit|   %.4f\n", dlogit_max);
    printf("  mean KL        %.6f (max %.6f)\n", kl_sum / n, kl_max);
    printf("  top-1 agree    %d / %d (%.1f%%)\n", agree, n, 100.0 * agree / n);
    printf("  greedy match   first %d of %d tokens identical\n", same, nf < nq ? nf : nq);
    printf("  speed          fp32 %.1f tok/s, int8 %.1f tok/s (%.2fx)\n",
           speed_f, speed_q, speed_q / speed_f);
    printf("\n--- fp32 ---\n");
    print_tokens(t, prompt_tokens[0], toks_f, nf);
    printf("\n--- int8 ---\n");
    print_tokens(t, prompt_tokens[0], toks_q, nq);

    free(toks_f); free(toks_q);
    model_free(&f); model_free(&q);
    return 0;
}
