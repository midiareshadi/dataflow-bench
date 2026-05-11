"""
Sparse SpMSpM — NVIDIA T4 Benchmark
Benchmarks all SpMSpM algorithms for all attention shapes and sparsity levels.
Run on Google Colab with T4 GPU runtime.
Results saved to sparse/results/t4/t4_results.csv
"""

import os
import sys
import time
import torch
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from workloads.llama7b_shapes import LLAMA7B_SHAPES
from sparse.sparsity.topk_mask import generate_sparse_pair, SPARSITY_LEVELS
from sparse.algorithms.vendor_baseline import spmm_vendor, verify_correctness as verify_vendor
from sparse.algorithms.inner_product import spmm_inner_product, verify_correctness as verify_inner
from sparse.algorithms.outer_product import spmm_outer_product, verify_correctness as verify_outer
from sparse.algorithms.gustavson_triton import spmm_gustavson, verify_correctness as verify_gustavson

ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 't4')
WARMUP_ITERS = 10
BENCH_ITERS  = 100

ALGORITHMS = [
    ('vendor',       spmm_vendor),
    ('inner_product', spmm_inner_product),
    ('outer_product', spmm_outer_product),
    ('gustavson',    spmm_gustavson),
]


def benchmark_fn(fn, A, B, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a function and return average latency in microseconds."""
    for _ in range(warmup):
        _ = fn(A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / iters) * 1e6


def run_correctness_checks(device):
    """Run correctness checks for all algorithms before benchmarking."""
    print("=" * 60)
    print("Running correctness checks for all algorithms...")
    print("=" * 60)

    # Use small shape for quick check
    M, K, N = 512, 128, 512
    sparsity = 0.90

    checks = [
        ('vendor',        verify_vendor),
        ('inner_product',  verify_inner),
        ('outer_product',  verify_outer),
        ('gustavson',     verify_gustavson),
    ]

    all_passed = True
    for name, verify_fn in checks:
        try:
            verify_fn(M, K, N, sparsity, device)
            print(f"  {name}: PASSED")
        except AssertionError as e:
            print(f"  {name}: FAILED — {e}")
            all_passed = False

    print()
    return all_passed


def run_benchmark(device):
    """Run full benchmark for all shapes, sparsity levels and algorithms."""
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {gpu_name}")
    print(f"Shapes: {len(ATTENTION_SHAPES)}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print(f"Algorithms: {[a[0] for a in ALGORITHMS]}")
    print(f"Expected rows: {len(ATTENTION_SHAPES) * len(SPARSITY_LEVELS) * len(ALGORITHMS)}")
    print("=" * 60)

    results = []

    for name, M, K, N, desc in ATTENTION_SHAPES:
        print(f"\nShape: {name} ({M}x{K}x{N}) — {desc}")

        for sparsity in SPARSITY_LEVELS:
            print(f"  sparsity={sparsity:.0%}")

            A_sparse, B_sparse, _, _ = generate_sparse_pair(
                M, K, N, sparsity,
                dtype=torch.float32,
                device=str(device)
            )

            # Get vendor latency for speedup calculation
            vendor_latency = None

            for algo_name, algo_fn in ALGORITHMS:
                try:
                    latency_us = benchmark_fn(algo_fn, A_sparse, B_sparse)

                    if algo_name == 'vendor':
                        vendor_latency = latency_us

                    speedup = (vendor_latency / latency_us
                               if vendor_latency and algo_name != 'vendor'
                               else 1.0)

                    results.append({
                        'hardware':    gpu_name,
                        'layer':       name,
                        'description': desc,
                        'M':           M,
                        'K':           K,
                        'N':           N,
                        'sparsity':    sparsity,
                        'algorithm':   algo_name,
                        'latency_us':  round(latency_us, 4),
                        'speedup_vs_vendor': round(speedup, 4),
                    })

                    print(f"    {algo_name:<15}: {latency_us:>10.2f} us"
                          + (f" | {speedup:.2f}x vs vendor"
                             if algo_name != 'vendor' else ""))

                except Exception as e:
                    print(f"    {algo_name:<15}: ERROR — {e}")

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, 't4_results.csv')
    df.to_csv(out_file, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_file}")
    print(f"Total rows: {len(df)} (expected "
          f"{len(ATTENTION_SHAPES) * len(SPARSITY_LEVELS) * len(ALGORITHMS)})")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required — run on T4 GPU runtime"
    device = torch.device("cuda")

    passed = run_correctness_checks(device)
    if not passed:
        print("Correctness checks failed — aborting benchmark")
        exit(1)

    run_benchmark(device)
