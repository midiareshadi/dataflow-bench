"""
Top-k Sparsity Mask Generator
Generates LLM-realistic attention sparsity masks using top-k per row strategy.

Background:
    Modern LLMs (H2O, SnapKV, StreamingLLM) show that attention score matrices
    are highly sparse — only top 5-30% of scores per row carry significant weight.
    Top-k sparsity keeps exactly k non-zeros per row, zeroing the rest.
    This is more structured than random sparsity and more realistic than block sparsity
    for LLM attention patterns.

Sparsity levels used:
    70% — moderate sparsity (H2O paper, production-realistic)
    90% — high sparsity (long context scenarios)
    95% — aggressive sparsity (StreamingLLM, very long context)
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from workloads.llama7b_shapes import LLAMA7B_SHAPES

# Attention score shapes from Phase 1
ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]

# LLM-realistic sparsity levels
SPARSITY_LEVELS = [0.70, 0.90, 0.95]


def generate_topk_mask(matrix: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Generate a top-k sparsity mask for a 2D matrix.
    Keeps the top (1-sparsity) fraction of values per row.
    All other positions are zeroed.

    Args:
        matrix:   2D tensor (M x N)
        sparsity: fraction of zeros (e.g. 0.90 = 90% zeros)

    Returns:
        Binary mask (1=keep, 0=zero), same shape as matrix
    """
    assert matrix.dim() == 2, "Input must be 2D tensor"
    assert 0.0 < sparsity < 1.0, "Sparsity must be between 0 and 1"

    M, N = matrix.shape
    k = max(1, int(N * (1.0 - sparsity)))  # number of non-zeros per row

    # Get top-k indices per row
    _, topk_indices = torch.topk(matrix.abs(), k, dim=1)

    # Build binary mask
    mask = torch.zeros_like(matrix)
    mask.scatter_(1, topk_indices, 1.0)

    return mask


def apply_topk_sparsity(matrix: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Apply top-k sparsity to a matrix.

    Args:
        matrix:   2D tensor (M x N)
        sparsity: fraction of zeros

    Returns:
        Sparse matrix with top-k values kept, rest zeroed
    """
    mask = generate_topk_mask(matrix, sparsity)
    return matrix * mask


def verify_sparsity(matrix: torch.Tensor, expected_sparsity: float, tol: float = 0.01):
    """Verify that a matrix has approximately the expected sparsity level."""
    actual_zeros = (matrix == 0).float().mean().item()
    assert abs(actual_zeros - expected_sparsity) < tol, \
        f"Expected sparsity {expected_sparsity:.2f}, got {actual_zeros:.2f}"


def generate_sparse_pair(M: int, K: int, N: int,
                          sparsity: float,
                          dtype: torch.dtype = torch.float32,
                          device: str = 'cpu') -> tuple:
    """
    Generate a pair of sparse matrices (A, B) for SpMSpM benchmarking.
    Both matrices use top-k sparsity — simulating sparse Q and K in attention.

    Args:
        M, K, N:  matrix dimensions (A: MxK, B: KxN)
        sparsity: sparsity level for both matrices
        dtype:    data type
        device:   compute device

    Returns:
        (A_sparse, B_sparse, A_csr, B_csr) — sparse tensors and CSR versions
    """
    # Generate random dense matrices
    A_dense = torch.randn(M, K, dtype=dtype, device=device)
    B_dense = torch.randn(K, N, dtype=dtype, device=device)

    # Apply top-k sparsity
    A_sparse = apply_topk_sparsity(A_dense, sparsity)
    B_sparse = apply_topk_sparsity(B_dense, sparsity)

    # Convert to CSR
    A_csr = A_sparse.to_sparse_csr()
    B_csr = B_sparse.to_sparse_csr()

    return A_sparse, B_sparse, A_csr, B_csr


def print_sparsity_stats(matrix: torch.Tensor, name: str = "Matrix"):
    """Print sparsity statistics for a matrix."""
    M, N = matrix.shape
    nnz = (matrix != 0).sum().item()
    total = M * N
    sparsity = 1.0 - nnz / total
    nnz_per_row = nnz / M

    print(f"{name}: shape=({M}x{N}), nnz={nnz}, "
          f"sparsity={sparsity:.1%}, nnz/row={nnz_per_row:.1f}")


def run_verification():
    """Verify top-k mask generation for all attention shapes and sparsity levels."""
    print("=" * 60)
    print("Verifying top-k sparsity mask generation")
    print("=" * 60)

    all_passed = True

    for name, M, K, N, desc in ATTENTION_SHAPES:
        print(f"\nShape: {name} ({M}x{K}x{N})")
        for sparsity in SPARSITY_LEVELS:
            A_sparse, B_sparse, A_csr, B_csr = generate_sparse_pair(
                M, K, N, sparsity
            )

            # Verify sparsity levels
            try:
                verify_sparsity(A_sparse, sparsity)
                verify_sparsity(B_sparse, sparsity)

                # Verify CSR conversion
                A_reconstructed = A_csr.to_dense()
                assert torch.allclose(A_sparse, A_reconstructed), \
                    "CSR reconstruction failed for A"

                print(f"  sparsity={sparsity:.0%} -> PASSED", end="  ")
                print_sparsity_stats(A_sparse, "A")

            except AssertionError as e:
                print(f"  sparsity={sparsity:.0%} -> FAILED: {e}")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All verification checks PASSED")
    else:
        print("Some checks FAILED — review output above")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
