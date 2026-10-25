#include "prof.h"

#include <string.h>
#include <time.h>

int prof_enabled = 0;
double prof_ms[PROF_CATEGORY_COUNT];

double prof_now_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1e6;
}

void prof_reset(void) { memset(prof_ms, 0, sizeof(prof_ms)); }
