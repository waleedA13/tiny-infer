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

// Computes rows first_row..end_row-1 of output = weights @ input. Each thread
// runs this same loop on its own rows, so every number comes out exactly the
// same as with one thread.
static void matmul_rows(float *output, const float *input, const float *weights,
                        int input_size, int first_row, int end_row) {
    for (int i = first_row; i < end_row; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < input_size; j++) row_sum += weights[(long)i * input_size + j] * input[j];
        output[i] = row_sum;
    }
}

// Thread pool. ops_init starts it once. Each matmul gives every thread a chunk
// of output rows, and the main thread does the first chunk itself.
//
// One matmul here only takes tens of microseconds, so putting threads to sleep
// and waking them up every time would cost more than we save. Instead the
// workers keep checking a job counter for a while, and only go to sleep if
// nothing shows up (like between tokens, or once generation is done).
#define SPIN_CHECKS_BEFORE_SLEEP 200000

static struct {
    int thread_count;           // counts the main thread too
    pthread_t *worker_threads;  // thread_count - 1 of them
    pthread_mutex_t sleep_lock;
    pthread_cond_t wake_signal;
    atomic_uint job_counter;    // goes up by one for each matmul
    atomic_int busy_workers;    // how many workers are still on the current job
    atomic_int should_stop;
    // The current job. Set before job_counter goes up, read by workers after.
    float *job_output;
    const float *job_input, *job_weights;
    int job_input_size, job_output_size;
} thread_pool = {
    .thread_count = 1,
    .sleep_lock = PTHREAD_MUTEX_INITIALIZER,
    .wake_signal = PTHREAD_COND_INITIALIZER,
};

static void run_rows_for_thread(int thread_index) {
    int total_rows = thread_pool.job_output_size;
    int rows_per_thread = (total_rows + thread_pool.thread_count - 1) / thread_pool.thread_count;
    int first_row = thread_index * rows_per_thread;
    int end_row = first_row + rows_per_thread;
    if (end_row > total_rows) end_row = total_rows;
    if (first_row < end_row)
        matmul_rows(thread_pool.job_output, thread_pool.job_input, thread_pool.job_weights,
                    thread_pool.job_input_size, first_row, end_row);
}

static void *worker_loop(void *thread_index_arg) {
    int thread_index = (int)(long)thread_index_arg;
    unsigned last_job_seen = 0;
    for (;;) {
        int spin_checks = 0;
        while (atomic_load_explicit(&thread_pool.job_counter, memory_order_acquire) == last_job_seen &&
               !atomic_load_explicit(&thread_pool.should_stop, memory_order_acquire)) {
            if (++spin_checks < SPIN_CHECKS_BEFORE_SLEEP) continue;
            pthread_mutex_lock(&thread_pool.sleep_lock);
            while (atomic_load(&thread_pool.job_counter) == last_job_seen &&
                   !atomic_load(&thread_pool.should_stop))
                pthread_cond_wait(&thread_pool.wake_signal, &thread_pool.sleep_lock);
            pthread_mutex_unlock(&thread_pool.sleep_lock);
        }
        if (atomic_load(&thread_pool.should_stop)) return NULL;
        last_job_seen = atomic_load(&thread_pool.job_counter);
        run_rows_for_thread(thread_index);
        atomic_fetch_sub_explicit(&thread_pool.busy_workers, 1, memory_order_release);
    }
}

int ops_init(int requested_threads) {
    if (requested_threads <= 0) {
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        requested_threads = cpu_count > 0 ? (int)cpu_count : 1;
    }
    thread_pool.worker_threads = malloc((size_t)requested_threads * sizeof(pthread_t));
    if (!thread_pool.worker_threads) return -1;
    thread_pool.thread_count = 1;
    for (int thread_index = 1; thread_index < requested_threads; thread_index++) {
        pthread_t *new_thread = &thread_pool.worker_threads[thread_index - 1];
        if (pthread_create(new_thread, NULL, worker_loop, (void *)(long)thread_index) != 0) {
            fprintf(stderr, "pthread_create failed, using %d thread(s)\n", thread_index);
            break;
        }
        thread_pool.thread_count = thread_index + 1;
    }
    return thread_pool.thread_count;
}

void ops_shutdown(void) {
    pthread_mutex_lock(&thread_pool.sleep_lock);
    atomic_store(&thread_pool.should_stop, 1);
    pthread_cond_broadcast(&thread_pool.wake_signal);
    pthread_mutex_unlock(&thread_pool.sleep_lock);
    for (int thread_index = 1; thread_index < thread_pool.thread_count; thread_index++)
        pthread_join(thread_pool.worker_threads[thread_index - 1], NULL);
    free(thread_pool.worker_threads);
    thread_pool.worker_threads = NULL;
    thread_pool.thread_count = 1;
    atomic_store(&thread_pool.should_stop, 0);
    atomic_store(&thread_pool.job_counter, 0);  // new workers start counting from 0
}

void matmul(float *out, const float *x, const float *w, int n, int d) {
    // This is where most of the time goes. Next up: SIMD in matmul_rows.
    if (thread_pool.thread_count == 1) { matmul_rows(out, x, w, n, 0, d); return; }

    thread_pool.job_output = out;
    thread_pool.job_input = x;
    thread_pool.job_weights = w;
    thread_pool.job_input_size = n;
    thread_pool.job_output_size = d;
    atomic_store_explicit(&thread_pool.busy_workers, thread_pool.thread_count - 1, memory_order_relaxed);
    // Do this while holding the lock so a worker about to sleep can't miss it.
    pthread_mutex_lock(&thread_pool.sleep_lock);
    atomic_fetch_add_explicit(&thread_pool.job_counter, 1, memory_order_release);
    pthread_cond_broadcast(&thread_pool.wake_signal);
    pthread_mutex_unlock(&thread_pool.sleep_lock);

    run_rows_for_thread(0);
    while (atomic_load_explicit(&thread_pool.busy_workers, memory_order_acquire) > 0) {}
}
