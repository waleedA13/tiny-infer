#ifndef PROF_H
#define PROF_H

// Per-op timers for --bench. Wrap a region in PROF_START/PROF_STOP and its time
// is added to prof_ms[category]. Costs one branch when profiling is off.

enum { PROF_MATMUL, PROF_ATTENTION, PROF_N };

extern int prof_enabled;
extern double prof_ms[PROF_N];

double prof_now_ms(void);
void   prof_reset(void);

#define PROF_START(start_ms) double start_ms = prof_enabled ? prof_now_ms() : 0.0
#define PROF_STOP(category, start_ms) do { if (prof_enabled) prof_ms[category] += prof_now_ms() - (start_ms); } while (0)

#endif
