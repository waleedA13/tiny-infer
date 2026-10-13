#!/usr/bin/env python3
"""Writes a checkpoint with the same header and tensor shapes as stories15M.bin,
filled with small deterministic random weights. Useful for benchmarking when the
real weights aren't available: the amount of work per token is identical."""
import array
import random
import struct
import sys

dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = 288, 768, 6, 6, 6, 32000, 256
head_size = dim // n_heads
kv_dim = n_kv_heads * head_size

counts = [
    vocab_size * dim,                 # token_embedding (shared with the classifier)
    n_layers * dim,                   # rms_att
    n_layers * dim * dim,             # wq
    n_layers * dim * kv_dim,          # wk
    n_layers * dim * kv_dim,          # wv
    n_layers * dim * dim,             # wo
    n_layers * dim,                   # rms_ffn
    n_layers * dim * hidden_dim,      # w1
    n_layers * hidden_dim * dim,      # w2
    n_layers * dim * hidden_dim,      # w3
    dim,                              # rms_final
    2 * seq_len * head_size,          # legacy freq_cis_real / freq_cis_imag
]

path = sys.argv[1] if len(sys.argv) > 1 else "models/synthetic15M.bin"
rng = random.Random(0)
with open(path, "wb") as f:
    # Positive vocab_size means the classifier shares the embedding, like stories15M.
    f.write(struct.pack("7i", dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len))
    for n in counts:
        f.write(array.array("f", (rng.uniform(-0.05, 0.05) for _ in range(n))).tobytes())
print(f"wrote {path} ({sum(counts) / 1e6:.1f}M params)")
