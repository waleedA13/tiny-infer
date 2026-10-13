#include "prof.h"

#include <string.h>
#include <time.h>

int prof_enabled = 0;
double prof_ms[PROF_N];

double prof_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void prof_reset(void) { memset(prof_ms, 0, sizeof(prof_ms)); }
