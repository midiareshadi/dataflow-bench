"""
Gustavson's Algorithm for SpMSpM — Triton Implementation (Step 2)

Algorithm (row-wise):
    For each row i of A:
        For each non-zero A[i,k]:
            C[i,:] += A[i,k] * B[k,:]

Key property exploited for Triton:
    Top-k sparsity gives UNIFORM row lengths in A.
    Every row has exactly NNZ_PER_ROW non-zeros.
    This makes the inner loop a compile-time constant — Triton friendly.

Kernel design:
    One program per (row tile of M, col tile of N)
    Inner loop: NNZ_PER_ROW iterations (compile-time constant)
    Each iteration: load A[i,k] and B[k,:], accumulate

Data layout:
    A: CSR format — efficient row access
    B: dense — efficient random row access B[k,:]
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
from sparse.algorithms.gustavson import spmm_gustavson_pytorch

ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]
WARMUP_ITERS = 10
BENCH_ITERS  = 100


@triton.jit
def gustavson_kernel(
    # A in CSR format
    A_values_ptr, A_col_indices_ptr, A_crow_indices_ptr,
    # B as dense matrix
    B_ptr,
    # Output C (dense)
    C_ptr,
    # Dimensions
    M, N, K,
    # Strides for B and C
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Non-zeros per row of A (uniform due to top-k)
    NNZ_PER_ROW: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Gustavson row-wise SpMSpM kernel.

    Each program computes a (BLOCK_M x BLOCK_N) tile of output C.

    For each row i in tile:
        For each non-zero A[i,k] (NNZ_PER_ROW iterations):
            C[i, n_offs] += A[i,k] * B[k, n_offs]

    This is the Gustavson algorithm:
        Row-wise processing with dense output accumulation.
        Exploits uniform row lengths for compile-time loop bounds.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Accumulator for output tile (BLOCK_M x BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load row pointers for this row tile
    crow = tl.load(A_crow_indices_ptr + m_offs, mask=m_mask, other=0)

    # Gustavson's row-wise loop:
    # For each non-zero position in each row of A (uniform = NNZ_PER_ROW)
    for nz in range(NNZ_PER_ROW):
        # Load column index k for this non-zero position
        # Shape: (BLOCK_M,) — one k per row
        a_col_ptrs = A_col_indices_ptr + crow + nz
        a_cols = tl.load(a_col_ptrs, mask=m_mask, other=0)

        # Load value A[i,k] for each row
        # Shape: (BLOCK_M,)
        a_val_ptrs = A_values_ptr + crow + nz
        a_vals = tl.load(a_val_ptrs, mask=m_mask, other=0.0)

        # Load B[k, n_offs] for each row's k
        # Shape: (BLOCK_M, BLOCK_N)
        b_ptrs = B_ptr + a_cols[:, None] * stride_bk + n_offs[None, :] * stride_bn
        b_rows = tl.load(b_ptrs,
                         mask=m_mask[:, None] & n_mask[None, :],
                         other=0.0)

        # Accumulate: C[i, n_offs] += A[i,k] * B[k, n_offs]
        # This is the Gustavson rank-1 update per row
        acc += a_vals[:, None] * b_rows

    # Write output tile
    c_ptrs = C_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    tl.store(c_ptrs, acc,
             mask=m_mask[:, None] & n_mask[None, :])


def spmm_gustavson(A_sparse: torch.Tensor,
                   B_sparse: torch.Tensor) -> torch.Tensor:
    """
    Gustavson SpMSpM Triton wrapper (Step 2).

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

    # Convert A to CSR
    A_csr = A_sparse.to_sparse_csr()
    A_crow = A_csr.crow_indices().to(torch.int32).contiguous()
    A_cols = A_csr.col_indices().to(torch.int32).contiguous()
    A_vals = A_csr.values().contiguous()

    # Get nnz per row (uniform due to top-k)
    nnz_per_row = int((A_crow[1] - A_crow[0]).item())

    C = torch.zeros(M, N, dtype=torch.float32, device=A_sparse.device)

    BLOCK_M, BLOCK_N = 32, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gustavson_kernel[grid](
        A_vals, A_cols, A_crow,
        B_sparse,
        C,
        M, N, K,
        B_sparse.stride(0), B_sparse.stride(1),
        C.stride(0), C.stride(1),
        NNZ_PER_ROW=nnz_per_row,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return C


def verify_correctness(M: int, K: int, N: int,
                        sparsity: float,
                        device: torch.device) -> bool:
    """Verify Gustavson Triton against PyTorch Step 1 and vendor baseline."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    ref     = spmm_vendor(A_sparse, B_sparse)
    step1   = spmm_gustavson_pytorch(A_sparse, B_sparse)
    triton_out = spmm_gustavson(A_sparse, B_sparse)

    # Check against vendor
    max_diff_vendor = (ref - triton_out).abs().max().item()
    # Check against Step 1
    max_diff_step1  = (step1 - triton_out).abs().max().item()

    assert max_diff_vendor < 1e-2, \
        f"Gustavson Triton vs vendor mismatch: {max_diff_vendor:.6f}"
    assert max_diff_step1 < 1e-2, \
        f"Gustavson Triton vs Step1 mismatch: {max_diff_step1:.6f}"
    return True


def benchmark(M: int, K: int, N: int,
              sparsity: float,
              device: torch.device,
              warmup: int = WARMUP_ITERS,
              iters: int = BENCH_ITERS) -> float:
    """Benchmark Gustavson Triton kernel."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    for _ in range(warmup):
        _ = spmm_gustavson(A_sparse, B_sparse)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = spmm_gustavson(A_sparse, B_sparse)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / iters) * 1e6


def run_correctness_checks(device: torch.device):
    """Run correctness checks for all shapes and sparsity levels."""
    print("=" * 60)
    print(f"Gustavson correctness checks (Step 2 - Triton) ({device})")
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
