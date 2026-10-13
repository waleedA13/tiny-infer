#!/usr/bin/env python3
"""Makes a fake checkpoint the same size and shape as stories15M.bin, filled with
small random numbers. Handy for benchmarking when you don't have the real weights,
since the model does the same amount of work either way."""
import array
import random
import struct
import sys

dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = 288, 768, 6, 6, 6, 32000, 256
head_size = dim // n_heads
kv_dim = n_kv_heads * head_size

tensor_sizes = [
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

output_path = sys.argv[1] if len(sys.argv) > 1 else "models/synthetic15M.bin"
random_generator = random.Random(0)
with open(output_path, "wb") as output_file:
    # Positive vocab_size means the classifier shares the embedding, like stories15M.
    output_file.write(struct.pack("7i", dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len))
    for tensor_size in tensor_sizes:
        weights = array.array("f", (random_generator.uniform(-0.05, 0.05) for _ in range(tensor_size)))
        output_file.write(weights.tobytes())
print(f"wrote {output_path} ({sum(tensor_sizes) / 1e6:.1f}M params)")
