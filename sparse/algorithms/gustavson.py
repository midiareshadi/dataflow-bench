"""
Gustavson's Algorithm for SpMSpM — Implementation

Algorithm (row-wise):
    For each row i of A:
        For each non-zero A[i,k]:
            For each non-zero B[k,j]:
                C[i,j] += A[i,k] * B[k,j]

Key characteristic:
    Processes output one row at a time.
    Accumulates contributions into a dense output row.
    Natural for CSR format — row access is efficient.

Implementation stages:
    Step 1: Pure PyTorch (CPU/GPU) — establishes correctness baseline
    Step 2: Triton kernel — only if Step 1 is correct and stable

This file implements Step 1 only.
Step 2 (Triton) will be added after Step 1 is verified.
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
from sparse.algorithms.vendor_baseline import spmm_vendor

ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]
WARMUP_ITERS = 5
BENCH_ITERS  = 50


def spmm_gustavson_pytorch(A_sparse: torch.Tensor,
                            B_sparse: torch.Tensor) -> torch.Tensor:
    """
    Gustavson's algorithm — pure PyTorch implementation (Step 1).

    Processes one row of A at a time.
    For each non-zero A[i,k], accumulates A[i,k] * B[k,:] into C[i,:].

    This is the row-wise SpMSpM algorithm from the GAMMA paper.
    Uses CSR format for A (efficient row access).
    Uses dense B for efficient row access B[k,:].

    Args:
        A_sparse: dense tensor with zeros (M x K)
        B_sparse: dense tensor with zeros (K x N)

    Returns:
        Dense output C (M x N)
    """
    M, K = A_sparse.shape
    K2, N = B_sparse.shape
    assert K == K2, "Incompatible dimensions"

    # Convert A to CSR for efficient row access
    A_csr = A_sparse.to_sparse_csr()
    crow = A_csr.crow_indices()
    cols = A_csr.col_indices()
    vals = A_csr.values()

    C = torch.zeros(M, N, dtype=torch.float32, device=A_sparse.device)

    # Gustavson's row-wise algorithm
    for i in range(M):
        row_start = crow[i].item()
        row_end   = crow[i + 1].item()

        # For each non-zero A[i,k]
        for nz in range(row_start, row_end):
            k   = cols[nz].item()
            val = vals[nz].item()

            # Accumulate val * B[k,:] into C[i,:]
            # This is the key operation: rank-1 update to output row
            C[i] += val * B_sparse[k]

    return C


def verify_correctness(M: int, K: int, N: int,
                        sparsity: float,
                        device: torch.device) -> bool:
    """Verify Gustavson against vendor baseline."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    ref = spmm_vendor(A_sparse, B_sparse)
    out = spmm_gustavson_pytorch(A_sparse, B_sparse)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-2, \
        f"Gustavson mismatch: max_diff={max_diff:.6f}"
    return True


def benchmark(M: int, K: int, N: int,
              sparsity: float,
              device: torch.device,
              warmup: int = WARMUP_ITERS,
              iters: int = BENCH_ITERS) -> float:
    """Benchmark Gustavson PyTorch implementation."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    # Warmup
    for _ in range(warmup):
        _ = spmm_gustavson_pytorch(A_sparse, B_sparse)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = spmm_gustavson_pytorch(A_sparse, B_sparse)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / iters) * 1e6


def run_correctness_checks(device: torch.device):
    """Run correctness checks for all shapes and sparsity levels."""
    print("=" * 60)
    print(f"Gustavson correctness checks (Step 1 - PyTorch) ({device})")
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
        print("All correctness checks PASSED — ready for Step 2 (Triton)")
    else:
        print("Some checks FAILED — fix before proceeding to Triton")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    run_correctness_checks(device)
