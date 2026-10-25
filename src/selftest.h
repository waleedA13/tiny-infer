#ifndef SELFTEST_H
#define SELFTEST_H

// Small checks for the math kernels. Add one every time you optimize a kernel,
// and compare the fast version against the naive one. Returns 0 if they all pass.
int run_selftest(void);

#endif
