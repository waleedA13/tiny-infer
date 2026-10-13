#ifndef PROF_H
#define PROF_H

// Per-op timers for --bench. Wrap a region in PROF_START/PROF_STOP and its time
// is added to prof_ms[category]. Costs one branch when profiling is off.

enum { PROF_MATMUL, PROF_ATTENTION, PROF_N };

extern int prof_enabled;
extern double prof_ms[PROF_N];

double prof_now_ms(void);
void   prof_reset(void);

#define PROF_START(var) double var = prof_enabled ? prof_now_ms() : 0.0
#define PROF_STOP(cat, var) do { if (prof_enabled) prof_ms[cat] += prof_now_ms() - (var); } while (0)

#endif
