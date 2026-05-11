"""
Output Stationary (OS) Matmul Kernel - Triton Implementation

Dataflow strategy:
- Outputs (matrix C) are the STATIONARY operand
- Each output tile accumulates ALL partial sums in registers
  before being written to memory ONCE at the end
- Both A (inputs) and B (weights) stream through
- Minimizes output/partial-sum movement — no intermediate writes

Tiling strategy:
- Large BLOCK_M and BLOCK_N: large output tiles stay in registers
- Large BLOCK_K: stream through K in large chunks to amortize
  the cost of loading both A and B
- Output tile shape: (BLOCK_M, BLOCK_N) — lives in registers entire time
- Both A and B stream through K dimension
"""

import triton
import triton.language as tl
import torch


@triton.jit
def matmul_os_kernel(
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
    Output Stationary kernel.
    Output tile (C) accumulates entirely in registers — never written
    to memory until all partial sums are complete.
    Both A and B stream through K dimension.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Output accumulator is STATIONARY in registers for entire K loop
    # This is the key distinction: partial sums NEVER leave registers
    # until the very end — no intermediate memory writes
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Both A and B stream through K dimension
    # OS: large BLOCK_K amortizes cost of loading both operands
    # since neither is stationary — we want each load to contribute
    # as much as possible to the output tile
    for k in range(0, K, BLOCK_K):
        # Both operands stream — neither is stationary
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)

        # Partial sum stays in register (output is stationary)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write output ONCE — only after all partial sums are accumulated
    # This is what makes it output stationary: single write per output element
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_os(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Output Stationary matmul wrapper.
    Tile sizes chosen to maximize output tile reuse:
    - Large BLOCK_M and BLOCK_N: large output tiles in registers
    - Large BLOCK_K: amortize cost of streaming both A and B
    """
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    K, N = B.shape
    C = torch.zeros((M, N), dtype=torch.float32, device=A.device)

    # OS tiling: large BLOCK_M, BLOCK_N, BLOCK_K
    # to maximize output tile accumulation in registers
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_os_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


def verify_correctness(M=256, K=256, N=256):
    """Verify OS kernel against torch.matmul."""
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    A_f = A.float()
    B_f = B.float()

    ref = torch.matmul(A_f, B_f)
    out = matmul_os(A_f, B_f)

    max_diff = (ref - out).abs().max().item()
    print(f"OS kernel max diff vs torch.matmul: {max_diff:.6f}")
    assert max_diff < 1e-2, f"Correctness check failed: max_diff={max_diff}"
    print("OS kernel correctness: PASSED")


if __name__ == '__main__':
    verify_correctness()
