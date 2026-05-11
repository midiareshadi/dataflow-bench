"""
Vendor Baseline — cuSPARSE (NVIDIA) / rocSPARSE (AMD)
SpMSpM using torch.sparse which calls vendor-optimized sparse libraries.

This serves as:
1. Ground truth correctness reference for all Triton implementations
2. Performance baseline — our algorithms aim to match or beat this
3. Hardware capability measurement — what is the best the hardware can do?

torch.sparse.mm() dispatch:
    NVIDIA GPU → cuSPARSE SpMM/SpMSpM
    AMD GPU    → rocSPARSE SpMM/SpMSpM
    CPU        → MKL sparse (Intel) or generic fallback
"""

import os
import sys
import time
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from workloads.llama7b_shapes import LLAMA7B_SHAPES
from sparse.sparsity.topk_mask import generate_sparse_pair, SPARSITY_LEVELS

# Attention score shapes only
ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]

WARMUP_ITERS = 10
BENCH_ITERS  = 100


def spmm_vendor(A_sparse: torch.Tensor,
                B_sparse: torch.Tensor) -> torch.Tensor:
    """
    SpMSpM using vendor-optimized library (cuSPARSE/rocSPARSE).
    Converts both matrices to CSR and uses torch.sparse.mm().

    On GPU: dispatches to cuSPARSE (NVIDIA) or rocSPARSE (AMD)
    On CPU: falls back to dense matmul (MKL SpMSpM not available on M1)

    Args:
        A_sparse: dense tensor with zeros (will be converted to sparse)
        B_sparse: dense tensor with zeros (will be converted to sparse)

    Returns:
        Dense output tensor C = A @ B
    """
    # CPU fallback — MKL SpMSpM not available on Apple M1
    if A_sparse.device.type == "cpu":
        return torch.matmul(A_sparse, B_sparse)

    # GPU path — cuSPARSE (NVIDIA) or rocSPARSE (AMD)
    A_csr = A_sparse.to_sparse_csr()
    B_csr = B_sparse.to_sparse_csr()

    C = torch.sparse.mm(A_csr, B_csr)

    # Return as dense
    if C.is_sparse or C.is_sparse_csr:
        return C.to_dense()
    return C


def verify_correctness(M: int, K: int, N: int,
                        sparsity: float,
                        device: torch.device) -> bool:
    """
    Verify vendor baseline against dense torch.matmul.

    Args:
        M, K, N:  matrix dimensions
        sparsity: sparsity level
        device:   compute device

    Returns:
        True if correct, raises AssertionError if not
    """
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    # Reference: dense matmul
    ref = torch.matmul(A_sparse, B_sparse)

    # Vendor sparse result
    out = spmm_vendor(A_sparse, B_sparse)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-3, \
        f"Vendor baseline mismatch: max_diff={max_diff:.6f}"
    return True


def benchmark_vendor(M: int, K: int, N: int,
                     sparsity: float,
                     device: torch.device,
                     warmup: int = WARMUP_ITERS,
                     iters: int = BENCH_ITERS) -> float:
    """
    Benchmark vendor SpMSpM and return average latency in microseconds.
    """
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    # Warmup
    for _ in range(warmup):
        _ = spmm_vendor(A_sparse, B_sparse)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = spmm_vendor(A_sparse, B_sparse)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / iters) * 1e6  # microseconds


def run_correctness_checks(device: torch.device):
    """Run correctness checks for all shapes and sparsity levels."""
    print("=" * 60)
    print(f"Vendor baseline correctness checks ({device})")
    print("=" * 60)

    all_passed = True
    for name, M, K, N, desc in ATTENTION_SHAPES:
        print(f"\nShape: {name} ({M}x{K}x{N})")
        for sparsity in SPARSITY_LEVELS:
            try:
                verify_correctness(M, K, N, sparsity, device)
                print(f"  sparsity={sparsity:.0%} -> PASSED")
            except AssertionError as e:
                print(f"  sparsity={sparsity:.0%} -> FAILED: {e}")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All correctness checks PASSED")
    else:
        print("Some checks FAILED")
    print("=" * 60)
    return all_passed


def run_benchmark(device: torch.device) -> list:
    """
    Run full vendor baseline benchmark.
    Returns list of result dicts.
    """
    print("\n" + "=" * 60)
    print(f"Vendor baseline benchmark ({device})")
    print("=" * 60)

    results = []

    for name, M, K, N, desc in ATTENTION_SHAPES:
        print(f"\nShape: {name} ({M}x{K}x{N})")
        for sparsity in SPARSITY_LEVELS:
            try:
                latency_us = benchmark_vendor(M, K, N, sparsity, device)
                nnz_A = int(M * K * (1 - sparsity))
                nnz_B = int(K * N * (1 - sparsity))
                flops = 2 * nnz_A  # approximate for SpMSpM

                results.append({
                    'algorithm':  'vendor',
                    'layer':      name,
                    'description': desc,
                    'M':          M,
                    'K':          K,
                    'N':          N,
                    'sparsity':   sparsity,
                    'latency_us': round(latency_us, 4),
                    'hardware':   str(device),
                })
                print(f"  sparsity={sparsity:.0%} -> {latency_us:.2f} us")

            except Exception as e:
                print(f"  sparsity={sparsity:.0%} -> ERROR: {e}")

    return results


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU found — using CPU")

    # Correctness first
    passed = run_correctness_checks(device)

    if passed:
        # Benchmark
        results = run_benchmark(device)
        print(f"\nCollected {len(results)} results")
        print(f"Expected: {len(ATTENTION_SHAPES) * len(SPARSITY_LEVELS)}")
