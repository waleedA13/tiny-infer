#!/usr/bin/env python3
"""Makes a fake checkpoint the same size and shape as stories15M.bin, filled with
small random numbers. Handy for benchmarking when you don't have the real weights,
since the model does the same amount of work either way."""
import array
import random
import struct
import sys

embed_dim, hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, max_seq_len = 288, 768, 6, 6, 6, 32000, 256
head_size = embed_dim // num_heads
kv_dim = num_kv_heads * head_size

tensor_sizes = [
    vocab_size * embed_dim,               # token_embedding (shared with the classifier)
    num_layers * embed_dim,               # attention_norm
    num_layers * embed_dim * embed_dim,   # query_weights
    num_layers * embed_dim * kv_dim,      # key_weights
    num_layers * embed_dim * kv_dim,      # value_weights
    num_layers * embed_dim * embed_dim,   # attention_out_weights
    num_layers * embed_dim,               # ffn_norm
    num_layers * embed_dim * hidden_dim,  # ffn_gate_weights
    num_layers * hidden_dim * embed_dim,  # ffn_down_weights
    num_layers * embed_dim * hidden_dim,  # ffn_up_weights
    embed_dim,                            # final_norm
    2 * max_seq_len * head_size,          # legacy freq_cis_real / freq_cis_imag
]

output_path = sys.argv[1] if len(sys.argv) > 1 else "models/synthetic15M.bin"
random_generator = random.Random(0)
with open(output_path, "wb") as output_file:
    # Positive vocab_size means the classifier shares the embedding, like stories15M.
    output_file.write(struct.pack("7i", embed_dim, hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, max_seq_len))
    for tensor_size in tensor_sizes:
        weights = array.array("f", (random_generator.uniform(-0.05, 0.05) for _ in range(tensor_size)))
        output_file.write(weights.tobytes())
print(f"wrote {output_path} ({sum(tensor_sizes) / 1e6:.1f}M params)")
