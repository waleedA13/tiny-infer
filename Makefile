CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
CPPFLAGS := -D_POSIX_C_SOURCE=200809L   # clock_gettime under -std=c11 on Linux
LDLIBS  := -lm -lpthread

SRC := src/main.c src/model.c src/tokenizer.c src/sampler.c src/ops.c src/prof.c
BIN := tiny-infer

$(BIN): $(SRC) src/*.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DBUILD_FLAGS='"$(CFLAGS)"' -o $@ $(SRC) $(LDLIBS)

# Fast build: native CPU features (NEON on Apple Silicon, AVX2 on x86) + OpenMP if available.
fast: CFLAGS := -O3 -march=native -ffast-math -Wall -std=c11
fast: $(BIN)

# Debug build with sanitizers. Run this before trusting any optimization.
debug: CFLAGS := -O0 -g -Wall -Wextra -std=c11 -fsanitize=address,undefined
debug: $(BIN)

test: $(BIN)
	./$(BIN) --selftest

# Benchmark. Always rebuilds its own binary so I never time an old debug build by mistake.
# You can override MODEL, BENCH_TOKENS or BENCH_RUNS.
MODEL        ?= models/stories15M.bin
BENCH_TOKENS ?= 200
BENCH_RUNS   ?= 5
BENCH_FLAGS  ?= -O3 -march=native -ffast-math -Wall -std=c11

bench: $(MODEL)
	$(CC) $(CPPFLAGS) $(BENCH_FLAGS) -DBUILD_FLAGS='"$(BENCH_FLAGS)"' -o $(BIN)-bench $(SRC) $(LDLIBS)
	./$(BIN)-bench --bench $(MODEL) -n $(BENCH_TOKENS) -r $(BENCH_RUNS)

# Fake weights the same size as stories15M. Speed doesn't depend on the actual
# numbers, so this does the same work without downloading anything.
models/synthetic15M.bin: scripts/make_synthetic_model.py
	python3 scripts/make_synthetic_model.py $@

clean:
	rm -f $(BIN) $(BIN)-bench

.PHONY: fast debug test bench clean
