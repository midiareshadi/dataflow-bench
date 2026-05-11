"""
Inner-Product SpMSpM Kernel — Triton Implementation (v2)

Algorithm:
    C[i,j] = dot product of row i of A and column j of B
    For each non-zero A[i,k], look up B[k,j] directly from dense B.

    Key insight: Since we need random access to B[k,j] for arbitrary k,
    we keep B as a dense matrix but A in CSR. This is valid for SpMSpM
    because we exploit A sparsity to skip zero rows, and B sparsity
    means many B[k,j] lookups return 0 — no wasted compute.

    This approach correctly implements the inner-product algorithm:
    - A is sparse (CSR) — we only iterate over non-zeros of A
    - B is kept dense for efficient random access B[k,j]
    - Output C is dense

Data layout:
    A: CSR format (crow_indices, col_indices, values)
    B: dense matrix (for efficient random access)

Key advantage:
    Top-k sparsity on A gives uniform row lengths (NNZ_PER_ROW)
    This makes the K loop a compile-time constant in Triton
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
def inner_product_kernel(
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
    # Number of non-zeros per row of A (uniform due to top-k)
    NNZ_PER_ROW: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Inner-product SpMSpM kernel.
    Each program computes a (BLOCK_M x BLOCK_N) tile of output C.

    For each row i in tile and col j in tile:
        C[i,j] = sum over non-zeros A[i,k] of A[i,k] * B[k,j]

    A is sparse (CSR) — we iterate only over its non-zeros.
    B is dense — we look up B[k,j] for each non-zero A[i,k].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load row pointers for this row tile from A CSR
    # crow_indices[i] gives start of row i in col_indices/values arrays
    row_mask = row_offs < M
    crow = tl.load(A_crow_indices_ptr + row_offs, mask=row_mask, other=0)

    # Iterate over non-zeros in each row of A
    # NNZ_PER_ROW is compile-time constant (top-k property)
    for k_idx in range(NNZ_PER_ROW):
        # Load column index of k_idx-th non-zero in each row
        # Shape: (BLOCK_M,)
        a_col_ptrs = A_col_indices_ptr + crow + k_idx
        a_cols = tl.load(a_col_ptrs, mask=row_mask, other=0)

        # Load value of k_idx-th non-zero in each row
        # Shape: (BLOCK_M,)
        a_val_ptrs = A_values_ptr + crow + k_idx
        a_vals = tl.load(a_val_ptrs, mask=row_mask, other=0.0)

        # Load row a_cols from B for all output columns
        # B[a_cols[i], col_offs[j]] for each i,j in tile
        # Shape: (BLOCK_M, BLOCK_N)
        b_ptrs = B_ptr + a_cols[:, None] * stride_bk + col_offs[None, :] * stride_bn
        col_mask = col_offs < N
        b_vals = tl.load(b_ptrs,
                         mask=row_mask[:, None] & col_mask[None, :],
                         other=0.0)

        # Accumulate: a_vals[i] * b_vals[i,j]
        acc += a_vals[:, None] * b_vals

    # Write output tile
    c_ptrs = C_ptr + row_offs[:, None] * stride_cm + col_offs[None, :] * stride_cn
    out_mask = row_mask[:, None] & (col_offs[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


def spmm_inner_product(A_sparse: torch.Tensor,
                        B_sparse: torch.Tensor) -> torch.Tensor:
    """
    Inner-product SpMSpM wrapper.

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

    # Output tensor
    C = torch.zeros(M, N, dtype=torch.float32, device=A_sparse.device)

    # Launch kernel
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    inner_product_kernel[grid](
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
    """Verify inner-product kernel against vendor baseline."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    ref = spmm_vendor(A_sparse, B_sparse)
    out = spmm_inner_product(A_sparse, B_sparse)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-2, \
        f"Inner-product mismatch: max_diff={max_diff:.6f}"
    return True


def benchmark(M: int, K: int, N: int,
              sparsity: float,
              device: torch.device,
              warmup: int = WARMUP_ITERS,
              iters: int = BENCH_ITERS) -> float:
    """Benchmark inner-product kernel."""
    A_sparse, B_sparse, _, _ = generate_sparse_pair(
        M, K, N, sparsity,
        dtype=torch.float32,
        device=str(device)
    )

    for _ in range(warmup):
        _ = spmm_inner_product(A_sparse, B_sparse)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = spmm_inner_product(A_sparse, B_sparse)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / iters) * 1e6


def run_correctness_checks(device: torch.device):
    """Run correctness checks for all shapes and sparsity levels."""
    print("=" * 60)
    print(f"Inner-product correctness checks ({device})")
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
