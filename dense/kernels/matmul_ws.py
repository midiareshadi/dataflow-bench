"""
Weight Stationary (WS) Matmul Kernel - Triton Implementation

Dataflow strategy:
- Weights (matrix B) are the STATIONARY operand
- Each output tile loads its corresponding weight tile ONCE into shared memory
- Weight tile stays fixed while input rows stream through
- Minimizes weight (filter) movement — matches how TPU MXU works

Tiling strategy:
- Large BLOCK_N: wide weight tiles maximize weight reuse per load
- Small BLOCK_K: weights loaded once per K-chunk, inputs stream
- Weight tile shape: (BLOCK_K, BLOCK_N) — loaded once, reused BLOCK_M times
- Input tile shape: (BLOCK_M, BLOCK_K) — streams through K dimension
"""

import triton
import triton.language as tl
import torch


@triton.jit
def matmul_ws_kernel(
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
    Weight Stationary kernel.
    B (weights) tile is loaded once into shared memory per output tile.
    A (inputs) streams through the K dimension.
    Partial sums accumulate in registers.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for this output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first K-block of A and B
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator lives in registers (output stationary within each PE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Stream through K dimension
    # WS: weight tile (B) is the stationary operand — it is the tile
    # that conceptually stays fixed in each PE's local register/scratchpad
    # while input rows (A) are streamed in
    for k in range(0, K, BLOCK_K):
        # Load weight tile (stationary operand) — large BLOCK_N maximizes reuse
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)

        # Load input tile (streaming operand)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)

        # Multiply-accumulate
        acc += tl.dot(a, b)

        # Advance pointers along K dimension
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write output
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_ws(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Weight Stationary matmul wrapper.
    Tile sizes chosen to maximize weight reuse:
    - Large BLOCK_N: wide weight tiles
    - BLOCK_K small relative to BLOCK_N
    """
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    K, N = B.shape
    C = torch.zeros((M, N), dtype=torch.float32, device=A.device)

    # WS tiling: large BLOCK_N to maximize weight tile reuse
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_ws_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


def verify_correctness(M=256, K=256, N=256):
    """Verify WS kernel against torch.matmul."""
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    A_f = A.float()
    B_f = B.float()

    ref = torch.matmul(A_f, B_f)
    out = matmul_ws(A_f, B_f)

    max_diff = (ref - out).abs().max().item()
    print(f"WS kernel max diff vs torch.matmul: {max_diff:.6f}")
    assert max_diff < 1e-2, f"Correctness check failed: max_diff={max_diff}"
    print("WS kernel correctness: PASSED")


if __name__ == '__main__':
    verify_correctness()
