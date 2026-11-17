"""
benchmark_fair.py — SpMSpM benchmark with a fair vendor baseline and a dense control.

Differences from benchmark_t4.py / benchmark_mi300x.py:

  1. The vendor path is timed in three parts instead of one.
     The original spmm_vendor() converted both operands to CSR and densified
     the result *inside* the timed region, so every iteration paid for format
     conversion that a real application does once (or never — the data would
     already be in CSR). Here conversion, the library call, and densification
     are measured separately, so the cost can be attributed.

  2. A dense torch.matmul baseline is included.
     The Triton "outer_product" kernel performs the full dense computation —
     it iterates over every k tile and never tests for zeros. Its true
     competitor is therefore cuBLAS/rocBLAS, not a sparse library. Without
     this control the comparison has no ceiling.

  3. Sparsity is swept to 99% to find the crossover.
     At 70-95% the dense path does 3-20x the useful FLOPs and still wins.
     The interesting question is where that stops being true.

  4. Each point is measured REPEATS times; median, min and max are recorded.
     Colab T4s are shared and throttle, and the original results differ by up
     to 3.7x between two scripts measuring the same configuration.

  5. Timing uses CUDA events rather than time.perf_counter().

Usage:
    python3 sparse/benchmark_fair.py              # all shapes
    python3 sparse/benchmark_fair.py --quick      # one shape, for a smoke test
"""
import argparse
import csv
import os
import statistics
import sys
import warnings

import torch

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from workloads.llama7b_shapes import LLAMA7B_SHAPES                    # noqa: E402
from sparse.sparsity.topk_mask import generate_sparse_pair             # noqa: E402
from sparse.algorithms.outer_product import spmm_outer_product         # noqa: E402
from sparse.algorithms.inner_product import spmm_inner_product         # noqa: E402
from sparse.algorithms.gustavson_triton import spmm_gustavson          # noqa: E402

ATTENTION_SHAPES = [s for s in LLAMA7B_SHAPES if "Attn" in s[0]]
SPARSITY_LEVELS = [0.70, 0.90, 0.95, 0.98, 0.99]

WARMUP = 10
ITERS = 50
REPEATS = 5


def time_us(fn, *args, warmup=WARMUP, iters=ITERS):
    """Median microseconds per call, measured with CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters     # ms -> us per call


def measure(name, fn, *args):
    """Run the timing REPEATS times and summarise."""
    samples = [time_us(fn, *args) for _ in range(REPEATS)]
    return {
        "algorithm": name,
        "latency_us": round(statistics.median(samples), 4),
        "min_us": round(min(samples), 4),
        "max_us": round(max(samples), 4),
        "spread": round(max(samples) / min(samples), 3),
    }


def run_point(M, K, N, sparsity, device):
    """All algorithms for one (shape, sparsity) point."""
    A, B, A_csr, B_csr = generate_sparse_pair(
        M, K, N, sparsity, dtype=torch.float32, device=str(device))

    rows = []

    # --- vendor, decomposed -------------------------------------------------
    # the library call alone, on data that is already in CSR
    rows.append(measure("vendor_prepared", lambda: torch.sparse.mm(A_csr, B_csr)))
    # what it costs to get there from dense
    rows.append(measure("vendor_convert", lambda: (A.to_sparse_csr(), B.to_sparse_csr())))
    # and what the original benchmark measured: convert + mm + densify
    def vendor_full():
        a = A.to_sparse_csr()
        b = B.to_sparse_csr()
        c = torch.sparse.mm(a, b)
        return c.to_dense() if (c.is_sparse or c.is_sparse_csr) else c
    rows.append(measure("vendor_full", vendor_full))

    # --- dense control ------------------------------------------------------
    rows.append(measure("dense_matmul", lambda: torch.matmul(A, B)))

    # --- the Triton kernels -------------------------------------------------
    rows.append(measure("outer_product", lambda: spmm_outer_product(A, B)))
    rows.append(measure("inner_product", lambda: spmm_inner_product(A, B)))
    rows.append(measure("gustavson", lambda: spmm_gustavson(A, B)))

    nnz_a = int((A != 0).sum().item())
    nnz_b = int((B != 0).sum().item())
    for r in rows:
        r.update(M=M, K=K, N=N, sparsity=sparsity,
                 nnz_A=nnz_a, nnz_B=nnz_b,
                 dense_flops=2 * M * K * N)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one shape only")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "a GPU is required"
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0) or "unknown"

    print("=" * 72)
    print(f"GPU      : {gpu}")
    print(f"torch    : {torch.__version__}")
    try:
        import triton
        print(f"triton   : {triton.__version__}")
    except Exception:
        print("triton   : unavailable")
    print(f"CUDA/HIP : {torch.version.cuda or torch.version.hip}")
    print(f"repeats  : {REPEATS} x {ITERS} iters, CUDA-event timed")
    print("=" * 72)

    shapes = ATTENTION_SHAPES[:1] if args.quick else ATTENTION_SHAPES
    out = []

    for name, M, K, N, desc in shapes:
        for sparsity in SPARSITY_LEVELS:
            print(f"\n{name} ({M}x{K}x{N})  sparsity={sparsity:.0%}")
            try:
                rows = run_point(M, K, N, sparsity, device)
            except Exception as e:                       # noqa: BLE001
                print(f"  ERROR: {e}")
                continue
            base = next(r for r in rows if r["algorithm"] == "vendor_prepared")
            for r in rows:
                r.update(layer=name, description=desc, hardware=gpu)
                speedup = base["latency_us"] / r["latency_us"]
                r["speedup_vs_vendor_prepared"] = round(speedup, 2)
                print(f"    {r['algorithm']:16s} {r['latency_us']:10.2f} us"
                      f"   x{speedup:7.2f} vs prepared vendor"
                      f"   (spread {r['spread']:.2f})")
            out.extend(rows)

    path = args.out or os.path.join(os.path.dirname(__file__), "results",
                                    "fair_results.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["hardware", "layer", "description", "M", "K", "N", "sparsity",
            "algorithm", "latency_us", "min_us", "max_us", "spread",
            "speedup_vs_vendor_prepared", "nnz_A", "nnz_B", "dense_flops"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nwrote {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
