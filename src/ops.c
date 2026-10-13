#define _POSIX_C_SOURCE 200809L

#include "ops.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void rmsnorm(float *out, const float *x, const float *weight, int n) {
    // TODO: ss = mean(x^2); scale = 1 / sqrt(ss + 1e-5); out[i] = weight[i] * x[i] * scale
    for (int i = 0; i < n; i++) out[i] = x[i] * weight[i];
}

void softmax(float *x, int n) {
    // TODO: subtract the max first for numerical stability, then exp and normalize.
    (void)x; (void)n;
}

// Rows [start, end) of out = W @ x. Every thread runs this same loop on its own
// rows, so each output is summed in the same order as the single-threaded build.
static void matmul_rows(float *out, const float *x, const float *w, int n, int start, int end) {
    for (int i = start; i < end; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += w[(long)i * n + j] * x[j];
        out[i] = sum;
    }
}

// Thread pool. Created once by ops_init; matmul hands each thread a contiguous
// slice of output rows and the calling thread does slice 0 itself.
//
// A matmul in this model is tens of microseconds, so waking sleeping threads
// through the kernel every call would eat the gain. Workers spin on a job
// counter for a while and only fall back to the condition variable when idle
// for longer (between tokens, or after generation ends).
#define SPIN_ITERS 200000

static struct {
    int n_threads;           // including the caller
    pthread_t *workers;      // n_threads - 1 of them
    pthread_mutex_t mu;
    pthread_cond_t cv;
    atomic_uint job;         // bumped once per matmul
    atomic_int pending;      // workers still running the current job
    atomic_int stop;
    // The current job. Written before job is bumped, read after.
    float *out;
    const float *x, *w;
    int n, d;
} pool = { .n_threads = 1, .mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER };

static void run_slice(int t) {
    int chunk = (pool.d + pool.n_threads - 1) / pool.n_threads;
    int start = t * chunk;
    int end = start + chunk < pool.d ? start + chunk : pool.d;
    if (start < end) matmul_rows(pool.out, pool.x, pool.w, pool.n, start, end);
}

static void *worker(void *arg) {
    int t = (int)(long)arg;
    unsigned seen = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&pool.job, memory_order_acquire) == seen &&
               !atomic_load_explicit(&pool.stop, memory_order_acquire)) {
            if (++spins < SPIN_ITERS) continue;
            pthread_mutex_lock(&pool.mu);
            while (atomic_load(&pool.job) == seen && !atomic_load(&pool.stop))
                pthread_cond_wait(&pool.cv, &pool.mu);
            pthread_mutex_unlock(&pool.mu);
        }
        if (atomic_load(&pool.stop)) return NULL;
        seen = atomic_load(&pool.job);
        run_slice(t);
        atomic_fetch_sub_explicit(&pool.pending, 1, memory_order_release);
    }
}

int ops_init(int n_threads) {
    if (n_threads <= 0) {
        long cpus = sysconf(_SC_NPROCESSORS_ONLN);
        n_threads = cpus > 0 ? (int)cpus : 1;
    }
    pool.workers = malloc((size_t)n_threads * sizeof(pthread_t));
    if (!pool.workers) return -1;
    pool.n_threads = 1;
    for (int t = 1; t < n_threads; t++) {
        if (pthread_create(&pool.workers[t - 1], NULL, worker, (void *)(long)t) != 0) {
            fprintf(stderr, "pthread_create failed, using %d thread(s)\n", t);
            break;
        }
        pool.n_threads = t + 1;
    }
    return pool.n_threads;
}

void ops_shutdown(void) {
    pthread_mutex_lock(&pool.mu);
    atomic_store(&pool.stop, 1);
    pthread_cond_broadcast(&pool.cv);
    pthread_mutex_unlock(&pool.mu);
    for (int t = 1; t < pool.n_threads; t++) pthread_join(pool.workers[t - 1], NULL);
    free(pool.workers);
    pool.workers = NULL;
    pool.n_threads = 1;
    atomic_store(&pool.stop, 0);
    atomic_store(&pool.job, 0);  // new workers start at seen = 0
}

void matmul(float *out, const float *x, const float *w, int n, int d) {
    // Hottest loop in the program. Next step: vectorize matmul_rows.
    if (pool.n_threads == 1) { matmul_rows(out, x, w, n, 0, d); return; }

    pool.out = out; pool.x = x; pool.w = w; pool.n = n; pool.d = d;
    atomic_store_explicit(&pool.pending, pool.n_threads - 1, memory_order_relaxed);
    // Bump under the lock so a worker that is about to sleep can't miss it.
    pthread_mutex_lock(&pool.mu);
    atomic_fetch_add_explicit(&pool.job, 1, memory_order_release);
    pthread_cond_broadcast(&pool.cv);
    pthread_mutex_unlock(&pool.mu);

    run_slice(0);
    while (atomic_load_explicit(&pool.pending, memory_order_acquire) > 0) {}
}
