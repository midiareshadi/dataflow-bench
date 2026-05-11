"""
Dense Dataflow - AMD MI300x Benchmark
Benchmarks WS, IS, OS Triton kernels for all LLaMA-7B matmul shapes.
Run this on AMD MI300x hardware with ROCm backend.
Results saved to dense/results/mi300x/mi300x_results.csv

Setup on MI300x:
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
    pip install triton
"""

import os
import sys
import time
import torch
import pandas as pd

# Make workloads and kernels importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))

from workloads.llama7b_shapes import LLAMA7B_SHAPES
from kernels.matmul_ws import matmul_ws, verify_correctness as verify_ws
from kernels.matmul_is import matmul_is, verify_correctness as verify_is
from kernels.matmul_os import matmul_os, verify_correctness as verify_os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'mi300x')
WARMUP_ITERS = 10
BENCH_ITERS  = 100

KERNELS = [
    ('ws', matmul_ws),
    ('is', matmul_is),
    ('os', matmul_os),
]


def benchmark_kernel(kernel_fn, A, B, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a kernel and return average latency in microseconds."""
    for _ in range(warmup):
        _ = kernel_fn(A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = kernel_fn(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / iters) * 1e6  # microseconds


def run_correctness_checks():
    print("=" * 60)
    print("Running correctness checks...")
    print("=" * 60)
    verify_ws()
    verify_is()
    verify_os()
    print("All correctness checks passed.\n")


def run_benchmark():
    """Run full benchmark for all shapes and dataflows on MI300x."""
    assert torch.cuda.is_available(), \
        "No GPU detected — ensure ROCm is installed and MI300x is visible"

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {gpu_name}")
    print(f"Shapes: {len(LLAMA7B_SHAPES)}")
    print(f"Dataflows: {len(KERNELS)}")
    print(f"Expected rows: {len(LLAMA7B_SHAPES) * len(KERNELS)}")
    print("=" * 60)

    run_correctness_checks()

    results = []

    for name, M, K, N, desc in LLAMA7B_SHAPES:
        print(f"\nShape: {name} ({M}x{K}x{N}) — {desc}")

        A = torch.randn(M, K, dtype=torch.float32, device=device)
        B = torch.randn(K, N, dtype=torch.float32, device=device)

        for df_name, kernel_fn in KERNELS:
            try:
                latency_us = benchmark_kernel(kernel_fn, A, B)
                flops = 2 * M * K * N
                tflops = flops / (latency_us * 1e-6) / 1e12

                results.append({
                    'hardware':    gpu_name,
                    'layer':       name,
                    'description': desc,
                    'M':           M,
                    'K':           K,
                    'N':           N,
                    'dataflow':    df_name.upper(),
                    'latency_us':  round(latency_us, 4),
                    'TFLOPS':      round(tflops, 4),
                })
                print(f"  {df_name.upper()}: {latency_us:.2f} us | {tflops:.2f} TFLOPS")

            except Exception as e:
                print(f"  {df_name.upper()}: ERROR — {e}")

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, 'mi300x_results.csv')
    df.to_csv(out_file, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_file}")
    print(f"Total rows: {len(df)} (expected {len(LLAMA7B_SHAPES) * len(KERNELS)})")
    print(f"{'='*60}")
    print(df[['layer', 'dataflow', 'latency_us', 'TFLOPS']].to_string())


if __name__ == '__main__':
    run_benchmark()
