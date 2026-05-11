"""
CSR Utility Functions
Provides CSR conversion, validation, and inspection tools
for sparse matrix operations in Phase 2.

CSR Format:
    crow_indices: shape (M+1,) — row pointers
    col_indices:  shape (nnz,) — column indices
    values:       shape (nnz,) — non-zero values

    Row i contains non-zeros at:
        col_indices[crow_indices[i] : crow_indices[i+1]]
        values[crow_indices[i] : crow_indices[i+1]]
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from workloads.llama7b_shapes import LLAMA7B_SHAPES
from sparse.sparsity.topk_mask import generate_sparse_pair, SPARSITY_LEVELS

# Attention score shapes
ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]


def dense_to_csr(matrix: torch.Tensor) -> tuple:
    """
    Convert a dense 2D tensor to CSR format.
    Returns (crow_indices, col_indices, values).

    Args:
        matrix: 2D dense tensor (M x N)

    Returns:
        crow_indices: (M+1,) row pointers
        col_indices:  (nnz,) column indices
        values:       (nnz,) non-zero values
    """
    assert matrix.dim() == 2, "Input must be 2D"
    csr = matrix.to_sparse_csr()
    return (
        csr.crow_indices().to(torch.int32),
        csr.col_indices().to(torch.int32),
        csr.values()
    )


def csr_to_dense(crow_indices: torch.Tensor,
                 col_indices: torch.Tensor,
                 values: torch.Tensor,
                 shape: tuple) -> torch.Tensor:
    """
    Convert CSR format back to dense tensor.

    Args:
        crow_indices: (M+1,) row pointers
        col_indices:  (nnz,) column indices
        values:       (nnz,) non-zero values
        shape:        (M, N) output shape

    Returns:
        Dense 2D tensor
    """
    M, N = shape
    csr = torch.sparse_csr_tensor(
        crow_indices.to(torch.int64),
        col_indices.to(torch.int64),
        values,
        size=(M, N)
    )
    return csr.to_dense()


def get_csr_stats(crow_indices: torch.Tensor,
                  col_indices: torch.Tensor,
                  values: torch.Tensor,
                  shape: tuple) -> dict:
    """
    Compute statistics for a CSR matrix.

    Returns dict with:
        nnz:           total non-zeros
        sparsity:      fraction of zeros
        nnz_per_row:   average non-zeros per row
        min_nnz_row:   minimum nnz in any row
        max_nnz_row:   maximum nnz in any row
        is_uniform:    True if all rows have same nnz (top-k property)
    """
    M, N = shape
    nnz = len(values)
    row_lengths = (crow_indices[1:] - crow_indices[:-1]).tolist()

    return {
        'nnz':         nnz,
        'sparsity':    1.0 - nnz / (M * N),
        'nnz_per_row': nnz / M,
        'min_nnz_row': min(row_lengths),
        'max_nnz_row': max(row_lengths),
        'is_uniform':  min(row_lengths) == max(row_lengths),
    }


def print_csr_stats(crow_indices, col_indices, values, shape, name="Matrix"):
    """Print CSR statistics in a readable format."""
    stats = get_csr_stats(crow_indices, col_indices, values, shape)
    print(f"{name} CSR stats:")
    print(f"  shape:       {shape}")
    print(f"  nnz:         {stats['nnz']}")
    print(f"  sparsity:    {stats['sparsity']:.1%}")
    print(f"  nnz/row:     {stats['nnz_per_row']:.1f}")
    print(f"  min/max nnz: {stats['min_nnz_row']} / {stats['max_nnz_row']}")
    print(f"  uniform rows:{stats['is_uniform']} (True = top-k structure)")


def verify_csr_correctness(dense: torch.Tensor,
                            crow_indices: torch.Tensor,
                            col_indices: torch.Tensor,
                            values: torch.Tensor) -> bool:
    """
    Verify that CSR representation matches original dense matrix.

    Returns True if correct, raises AssertionError if not.
    """
    M, N = dense.shape
    reconstructed = csr_to_dense(crow_indices, col_indices, values, (M, N))
    assert torch.allclose(dense, reconstructed, atol=1e-6), \
        f"CSR reconstruction mismatch: max diff = {(dense - reconstructed).abs().max()}"
    return True


def run_verification():
    """Verify CSR utilities for all attention shapes and sparsity levels."""
    print("=" * 60)
    print("Verifying CSR utilities")
    print("=" * 60)

    all_passed = True

    for name, M, K, N, desc in ATTENTION_SHAPES:
        print(f"\nShape: {name} ({M}x{K}x{N})")
        for sparsity in SPARSITY_LEVELS:
            A_sparse, B_sparse, A_csr, B_csr = generate_sparse_pair(
                M, K, N, sparsity
            )

            try:
                # Convert to CSR
                crow, cols, vals = dense_to_csr(A_sparse)

                # Verify correctness
                verify_csr_correctness(A_sparse, crow, cols, vals)

                # Get stats
                stats = get_csr_stats(crow, cols, vals, (M, K))

                # Verify uniform rows (top-k property)
                assert stats['is_uniform'], \
                    "Top-k sparsity should produce uniform row lengths"

                print(f"  sparsity={sparsity:.0%} -> PASSED | "
                      f"nnz={stats['nnz']} | "
                      f"nnz/row={stats['nnz_per_row']:.0f} | "
                      f"uniform={stats['is_uniform']}")

            except AssertionError as e:
                print(f"  sparsity={sparsity:.0%} -> FAILED: {e}")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All CSR verification checks PASSED")
    else:
        print("Some checks FAILED — review output above")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
