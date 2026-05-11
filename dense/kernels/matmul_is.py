"""
Input Stationary (IS) Matmul Kernel - Triton Implementation

Dataflow strategy:
- Inputs (matrix A) are the STATIONARY operand
- Each input tile is loaded ONCE into shared memory and reused
  across multiple weight tiles
- Weights (B) stream through while input stays fixed
- Minimizes input activation movement

Tiling strategy:
- Large BLOCK_M: tall input tiles maximize input reuse per load
- Small BLOCK_K: inputs loaded once per K-chunk, weights stream
- Input tile shape: (BLOCK_M, BLOCK_K) — loaded once, reused BLOCK_N times
- Weight tile shape: (BLOCK_K, BLOCK_N) — streams through N dimension
"""

import triton
import triton.language as tl
import torch


@triton.jit
def matmul_is_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Input Stationary kernel.
    A (inputs) tile is loaded once into shared memory per output tile row.
    B (weights) streams through the N dimension.
    Partial sums accumulate in registers.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Stream through K dimension
    # IS: input tile (A) is the stationary operand — it conceptually
    # stays fixed in each PE's scratchpad while weight columns (B)
    # stream through. Large BLOCK_M maximizes input row reuse.
    for k in range(0, K, BLOCK_K):
        # Load input tile (stationary operand) — large BLOCK_M maximizes reuse
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)

        # Load weight tile (streaming operand)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_is(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Input Stationary matmul wrapper.
    Tile sizes chosen to maximize input reuse:
    - Large BLOCK_M: tall input tiles
    - BLOCK_K small relative to BLOCK_M
    """
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    K, N = B.shape
    C = torch.zeros((M, N), dtype=torch.float32, device=A.device)

    # IS tiling: large BLOCK_M to maximize input tile reuse
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_is_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


def verify_correctness(M=256, K=256, N=256):
    """Verify IS kernel against torch.matmul."""
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    A_f = A.float()
    B_f = B.float()

    ref = torch.matmul(A_f, B_f)
    out = matmul_is(A_f, B_f)

    max_diff = (ref - out).abs().max().item()
    print(f"IS kernel max diff vs torch.matmul: {max_diff:.6f}")
    assert max_diff < 1e-2, f"Correctness check failed: max_diff={max_diff}"
    print("IS kernel correctness: PASSED")


if __name__ == '__main__':
    verify_correctness()
