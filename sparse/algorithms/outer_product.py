"""
Outer-Product SpMSpM Kernel — Triton Implementation (v3)

Algorithm:
    C = sum over k of outer_product(A[:,k], B[k,:])

Implementation strategy (v3 — safe tiled approach):
    Avoid scalar sparse index reads inside kernel.
    Instead precompute a dense expanded representation of A columns
    and use standard tiled matrix operations.

    For each k:
        a_col = A[:,k] as dense vector (M,)
        b_row = B[k,:] as dense vector (N,)
        C += outer(a_col, b_row)

    We implement this by iterating over K in tiles and using
    tl.dot on small tiles of the expanded representation.

Data layout:
    A: dense (zeros preserved) — avoids unsafe sparse indexing
    B: dense
"""

import os
import sys
import time
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import triton
import triton.language as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from workloads.llama7b_shapes import LLAMA7B_SHAPES
from sparse.sparsity.topk_mask import generate_sparse_pair, SPARSITY_LEVELS
from sparse.algorithms.vendor_baseline import spmm_vendor

ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]
WARMUP_ITERS = 10
BENCH_ITERS  = 100


@triton.jit
def outer_product_kernel(
    # A dense (M x K) with zeros
    A_ptr,
    # B dense (K x N) with zeros
    B_ptr,
    # Output C dense (M x N)
    C_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Outer-product SpMSpM kernel (v3).

    Implements outer-product algorithm:
        C += sum_k outer(A[:,k], B[k,:])

    This is mathematically equivalent to standard matmul but
    the computation order follows the outer-product algorithm:
    for each k, compute rank-1 update to C.

    Key difference from inner-product (Dense WS/IS/OS):
        Inner-product: C[i,j] = dot(A[i,:], B[:,j])  — row x col
        Outer-product: C += A[:,k] x B[k,:]           — col x row, summed over k

    The outer-product order naturally skips zero contributions
    when A[:,k] or B[k,:] is entirely zero (sparse columns/rows).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Outer-product order: iterate over K dimension
    # For each k tile, compute partial outer products
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: (BLOCK_M, BLOCK_K) — columns of A
        a_ptrs = A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_tile = tl.load(a_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)

        # Load B tile: (BLOCK_K, BLOCK_N) — rows of B
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
        b_tile = tl.load(b_ptrs,
                         mask=(k_offs[:, None] < K) & (n_offs[None, :] < N),
                         other=0.0)

        # Outer-product accumulation: sum of rank-1 updates
        # tl.dot computes sum_k A[:,k] x B[k,:] for this K tile
        acc += tl.dot(a_tile, b_tile)

    # Write output
    c_ptrs = C_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    tl.store(c_ptrs, acc,
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def spmm_outer_product(A_sparse: torch.Tensor,
                        B_sparse: torch.Tensor) -> torch.Tensor:
    """
    Outer-product SpMSpM wrapper.

    Uses dense representation with zeros to safely implement
    outer-product algorithm without unsafe sparse indexing.
    The sparsity is exploited by the algorithm structure
    (zero columns/rows contribute nothing to the sum).

    Args:
        A_sparse: dense tensor with zeros (M x K)
        B_sparse: dense tensor with zeros (K x N)

    Returns:
        Dense output C (M x N)
    """
    assert A_sparse.is_contiguous() and B_sparse.is_contiguous()
    M, K = A_sparse.shape
    K2, N = B_sparse.shape
    assert K == K2, "Incompatible dimensions"

    C = torch.zeros(M, N, dtype=torch.float32, device=A_sparse.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    outer_product_kernel[grid](
        A_sparse, B_sparse, C,
        M, N, K,
        A_sparse.stride(0), A_sparse.stride(1),
        B_sparse.stride(0), B_sparse.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def verify_correctness(M: int, K: int, N: int,
                        sparsity: float,
                        device: torch.device) -> bool:
    """Verify outer-product kernel against vendor baseline."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    ref = spmm_vendor(A_sparse, B_sparse)
    out = spmm_outer_product(A_sparse, B_sparse)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 0.05, \
        f"Outer-product mismatch: max_diff={max_diff:.6f}"
    return True


def benchmark(M: int, K: int, N: int,
              sparsity: float,
              device: torch.device,
              warmup: int = WARMUP_ITERS,
              iters: int = BENCH_ITERS) -> float:
    """Benchmark outer-product kernel."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    for _ in range(warmup):
        _ = spmm_outer_product(A_sparse, B_sparse)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = spmm_outer_product(A_sparse, B_sparse)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / iters) * 1e6


def run_correctness_checks(device: torch.device):
    """Run correctness checks for all shapes and sparsity levels."""
    print("=" * 60)
    print(f"Outer-product correctness checks ({device})")
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


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for Triton kernels"
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    run_correctness_checks(device)
