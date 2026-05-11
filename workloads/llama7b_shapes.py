"""
LLaMA-7B matrix multiplication shapes for dataflow benchmarking.

Architecture parameters (from LLaMA-7B config):
    Hidden dim:              4096
    FFN intermediate dim:    11008
    Number of attention heads: 32
    Head dim:                128  (= 4096 / 32)
    Sequence lengths tested: 512, 1024, 2048

Each shape is defined as (M, K, N) representing:
    A: (M x K)
    B: (K x N)
    C: (M x N)  <- output
"""

LLAMA7B_SHAPES = [
    # (name, M, K, N, description)

    # --- Attention projection layers ---
    # All four projections share the same shape in LLaMA-7B
    ("M1_Q_proj",           4096,  4096,  4096,  "Query projection"),
    ("M2_K_proj",           4096,  4096,  4096,  "Key projection"),
    ("M3_V_proj",           4096,  4096,  4096,  "Value projection"),
    ("M4_Out_proj",         4096,  4096,  4096,  "Attention output projection"),

    # --- Feed-forward network layers ---
    ("M5_FFN_gate",         4096,  11008, 4096,  "FFN gate projection (SwiGLU)"),
    ("M6_FFN_up",           4096,  11008, 4096,  "FFN up projection (SwiGLU)"),
    ("M7_FFN_down",         4096,  4096,  11008, "FFN down projection"),

    # --- Attention score computation: Q x K^T ---
    # Shape: (seq_len, head_dim) x (head_dim, seq_len)
    ("M8_AttnScore_512",    512,   128,   512,   "Q x K^T, seq_len=512"),
    ("M9_AttnScore_1024",   1024,  128,   1024,  "Q x K^T, seq_len=1024"),
    ("M10_AttnScore_2048",  2048,  128,   2048,  "Q x K^T, seq_len=2048"),

    # --- Attention output computation: scores x V ---
    # Shape: (seq_len, seq_len) x (seq_len, head_dim)
    ("M11_AttnOut_512",     512,   512,   128,   "scores x V, seq_len=512"),
    ("M12_AttnOut_1024",    1024,  1024,  128,   "scores x V, seq_len=1024"),
    ("M13_AttnOut_2048",    2048,  2048,  128,   "scores x V, seq_len=2048"),
]


def get_shape(name):
    """Return (M, K, N) for a given shape name."""
    for entry in LLAMA7B_SHAPES:
        if entry[0] == name:
            return entry[1], entry[2], entry[3]
    raise ValueError(f"Shape '{name}' not found.")


def print_summary():
    """Print a formatted summary of all shapes."""
    print(f"{'Name':<25} {'M':>6} {'K':>6} {'N':>6}  Description")
    print("-" * 70)
    for name, M, K, N, desc in LLAMA7B_SHAPES:
        print(f"{name:<25} {M:>6} {K:>6} {N:>6}  {desc}")


if __name__ == "__main__":
    print_summary()
