CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDLIBS  := -lm -lpthread

SRC := src/main.c src/model.c src/tokenizer.c src/sampler.c src/ops.c src/compare.c
BIN := tiny-infer

$(BIN): $(SRC) src/*.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# Fast build: native CPU features (NEON on Apple Silicon, AVX2 on x86) + OpenMP if available.
fast: CFLAGS := -O3 -march=native -ffast-math -Wall -std=c11
fast: $(BIN)

# Debug build with sanitizers. Run this before trusting any optimization.
debug: CFLAGS := -O0 -g -Wall -Wextra -std=c11 -fsanitize=address,undefined
debug: $(BIN)

test: $(BIN)
	./$(BIN) --selftest

clean:
	rm -f $(BIN)

.PHONY: fast debug test clean
