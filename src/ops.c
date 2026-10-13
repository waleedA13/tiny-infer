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

// Computes rows start..end-1 of out = W @ x. Each thread runs this same loop on
// its own rows, so every number comes out exactly the same as with one thread.
static void matmul_rows(float *out, const float *x, const float *w, int n, int start, int end) {
    for (int i = start; i < end; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += w[(long)i * n + j] * x[j];
        out[i] = sum;
    }
}

// Thread pool. ops_init starts it once. Each matmul gives every thread a chunk
// of output rows, and the main thread does the first chunk itself.
//
// One matmul here only takes tens of microseconds, so putting threads to sleep
// and waking them up every time would cost more than we save. Instead the
// workers keep checking a job counter for a while, and only go to sleep if
// nothing shows up (like between tokens, or once generation is done).
#define SPIN_ITERS 200000

static struct {
    int n_threads;           // counts the main thread too
    pthread_t *workers;      // n_threads - 1 of them
    pthread_mutex_t mu;
    pthread_cond_t cv;
    atomic_uint job;         // goes up by one for each matmul
    atomic_int pending;      // how many workers are still busy
    atomic_int stop;
    // The current job. Set before job goes up, read by workers after.
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
    atomic_store(&pool.job, 0);  // new workers start counting from 0
}

void matmul(float *out, const float *x, const float *w, int n, int d) {
    // This is where most of the time goes. Next up: SIMD in matmul_rows.
    if (pool.n_threads == 1) { matmul_rows(out, x, w, n, 0, d); return; }

    pool.out = out; pool.x = x; pool.w = w; pool.n = n; pool.d = d;
    atomic_store_explicit(&pool.pending, pool.n_threads - 1, memory_order_relaxed);
    // Do this while holding the lock so a worker about to sleep can't miss it.
    pthread_mutex_lock(&pool.mu);
    atomic_fetch_add_explicit(&pool.job, 1, memory_order_release);
    pthread_cond_broadcast(&pool.cv);
    pthread_mutex_unlock(&pool.mu);

    run_slice(0);
    while (atomic_load_explicit(&pool.pending, memory_order_acquire) > 0) {}
}
