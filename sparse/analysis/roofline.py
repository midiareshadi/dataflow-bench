"""
Bandwidth Utilization Analysis
Measures achieved memory bandwidth for each SpMSpM algorithm.
Answers: "Why does outer-product win?"

Method:
    For our Triton kernels (inner, outer, Gustavson):
        bytes_accessed = dense storage (A + B + C)
        These kernels read full dense matrices — honest measurement

    For vendor (cuSPARSE/rocSPARSE):
        bytes_accessed = sparse storage (nnz_A + nnz_B + output)
        Vendor reads only non-zeros — honest measurement

Peak bandwidth:
    T4:     300 GB/s
    MI300x: 5300 GB/s

Run on: Google Colab T4 and AMD MI300x
"""

import os
import sys
import time
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from workloads.llama7b_shapes import LLAMA7B_SHAPES
from sparse.sparsity.topk_mask import generate_sparse_pair, SPARSITY_LEVELS
from sparse.algorithms.outer_product import spmm_outer_product
from sparse.algorithms.inner_product import spmm_inner_product
from sparse.algorithms.gustavson_triton import spmm_gustavson
from sparse.algorithms.vendor_baseline import spmm_vendor

ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if 'Attn' in s[0]]
WARMUP_ITERS = 10
BENCH_ITERS  = 100

# Peak memory bandwidth in GB/s
PEAK_BW = {
    'T4':     300.0,
    'MI300x': 5300.0,
}


def get_hardware_name():
    """Get simplified hardware name for peak BW lookup."""
    name = torch.cuda.get_device_name(0)
    if 'T4' in name:
        return 'T4'
    elif 'MI300' in name:
        return 'MI300x'
    else:
        # Try ROCm device name
        try:
            import subprocess
            result = subprocess.run(['rocm-smi', '--showproductname'],
                                   capture_output=True, text=True)
            if 'MI300' in result.stdout:
                return 'MI300x'
        except Exception:
            pass
        # Default fallback
        if not name.strip():
            return 'MI300x'  # empty name on ROCm = MI300x
        return name


def benchmark_fn(fn, A, B, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark and return average latency in seconds."""
    for _ in range(warmup):
        _ = fn(A, B)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters


def compute_dense_bytes(A, B, C_shape):
    """
    Bytes accessed by dense-storage kernels (inner, outer, Gustavson).
    These kernels store and read full dense matrices.
    """
    bytes_A = A.numel() * A.element_size()
    bytes_B = B.numel() * B.element_size()
    bytes_C = C_shape[0] * C_shape[1] * 4  # float32 output
    return bytes_A + bytes_B + bytes_C


def compute_sparse_bytes(A_sparse, B_sparse, C_shape, sparsity):
    """
    Bytes accessed by vendor CSR kernel.
    Vendor reads only non-zeros from A and B.
    """
    M, K = A_sparse.shape
    K2, N = B_sparse.shape
    nnz_A = int(M * K * (1 - sparsity))
    nnz_B = int(K * N * (1 - sparsity))
    # CSR: values (float32) + col_indices (int32) per non-zero
    #      + row_pointers (int32) per row
    bytes_A = nnz_A * (4 + 4) + (M + 1) * 4
    bytes_B = nnz_B * (4 + 4) + (K + 1) * 4
    bytes_C = C_shape[0] * C_shape[1] * 4  # dense output
    return bytes_A + bytes_B + bytes_C


def run_bandwidth_analysis(device):
    """Run bandwidth utilization analysis for all shapes and sparsity levels."""
    hw_name = get_hardware_name()
    peak_bw = PEAK_BW.get(hw_name, 300.0)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Peak bandwidth: {peak_bw} GB/s")
    print("=" * 80)

    results = []

    algorithms = [
        ('vendor',        lambda a, b: spmm_vendor(a, b),        'sparse'),
        ('inner_product', lambda a, b: spmm_inner_product(a, b), 'dense'),
        ('outer_product', lambda a, b: spmm_outer_product(a, b), 'dense'),
        ('gustavson',     lambda a, b: spmm_gustavson(a, b),     'dense'),
    ]

    for name, M, K, N, desc in ATTENTION_SHAPES:
        print(f"\nShape: {name} ({M}x{K}x{N})")

        for sparsity in SPARSITY_LEVELS:
            A_sparse, B_sparse, _, _ = generate_sparse_pair(
                M, K, N, sparsity,
                dtype=torch.float32,
                device=str(device)
            )

            print(f"  sparsity={sparsity:.0%}")

            for algo_name, algo_fn, storage_type in algorithms:
                try:
                    # Measure latency
                    latency_s = benchmark_fn(algo_fn, A_sparse, B_sparse)
                    latency_us = latency_s * 1e6

                    # Compute bytes accessed
                    C_shape = (M, N)
                    if storage_type == 'dense':
                        bytes_accessed = compute_dense_bytes(
                            A_sparse, B_sparse, C_shape)
                    else:
                        bytes_accessed = compute_sparse_bytes(
                            A_sparse, B_sparse, C_shape, sparsity)

                    # Compute bandwidth metrics
                    achieved_bw = bytes_accessed / latency_s / 1e9  # GB/s
                    bw_utilization = achieved_bw / peak_bw * 100    # %

                    results.append({
                        'hardware':       hw_name,
                        'layer':          name,
                        'M':              M,
                        'K':              K,
                        'N':              N,
                        'sparsity':       sparsity,
                        'algorithm':      algo_name,
                        'storage_type':   storage_type,
                        'latency_us':     round(latency_us, 4),
                        'bytes_accessed': bytes_accessed,
                        'achieved_bw_GBs': round(achieved_bw, 2),
                        'peak_bw_GBs':    peak_bw,
                        'bw_utilization': round(bw_utilization, 2),
                    })

                    print(f"    {algo_name:<15}: "
                          f"{latency_us:>10.2f} us | "
                          f"BW={achieved_bw:>8.2f} GB/s | "
                          f"util={bw_utilization:>6.2f}%")

                except Exception as e:
                    print(f"    {algo_name:<15}: ERROR — {e}")

    return results


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    results = run_bandwidth_analysis(device)

    import pandas as pd
    df = pd.DataFrame(results)

    # Summary: mean bandwidth utilization per algorithm
    print("\n" + "=" * 80)
    print("Bandwidth Utilization Summary")
    print("=" * 80)
    summary = df.groupby(['algorithm', 'storage_type'])[
        ['achieved_bw_GBs', 'bw_utilization']].mean().round(2)
    print(summary)

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    hw = df['hardware'].iloc[0].lower().replace(' ', '_')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{hw}_bandwidth.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
