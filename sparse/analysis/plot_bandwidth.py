"""
Bandwidth Utilization Plot
Generates figure showing achieved memory bandwidth per algorithm per hardware.
Answers visually: "Why does outer-product win?"
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    'vendor':        '#9E9E9E',
    'inner_product': '#2196F3',
    'outer_product': '#4CAF50',
    'gustavson':     '#FF9800',
}

ALGO_LABELS = {
    'vendor':        'Vendor\n(cuSPARSE/rocSPARSE)',
    'inner_product': 'Inner-Product\n(Triton)',
    'outer_product': 'Outer-Product\n(Triton)',
    'gustavson':     'Gustavson\n(Triton)',
}

PEAK_BW = {'T4': 300.0, 'MI300x': 5300.0}
ALGOS   = ['vendor', 'inner_product', 'outer_product', 'gustavson']


def load_bandwidth_data():
    """Load bandwidth CSV files for both hardware."""
    results = []
    for hw in ['t4', 'mi300x']:
        path = os.path.join(RESULTS_DIR, f'{hw}_bandwidth.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            results.append(df)
    if not results:
        raise FileNotFoundError("No bandwidth CSV files found in results/")
    return pd.concat(results, ignore_index=True)


def plot_bandwidth_utilization(df):
    """
    Figure 6: Mean bandwidth utilization per algorithm per hardware.
    Shows why outer-product wins — it achieves highest bandwidth utilization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Memory Bandwidth Utilization per Algorithm\n' +
        '(Higher = more efficient use of GPU memory bandwidth)',
        fontsize=13, fontweight='bold'
    )

    for ax, hw in zip(axes, ['T4', 'MI300x']):
        hw_df = df[df['hardware'] == hw]
        if hw_df.empty:
            ax.set_title(f'{hw} — no data')
            continue

        peak = PEAK_BW[hw]

        # Mean bandwidth per algorithm
        means = []
        stds  = []
        for algo in ALGOS:
            algo_df = hw_df[hw_df['algorithm'] == algo]
            means.append(algo_df['achieved_bw_GBs'].mean())
            stds.append(algo_df['achieved_bw_GBs'].std())

        x = np.arange(len(ALGOS))
        bars = ax.bar(x, means, color=[COLORS[a] for a in ALGOS],
                      alpha=0.85, yerr=stds, capsize=4)

        # Add peak bandwidth line
        ax.axhline(y=peak, color='red', linestyle='--',
                   linewidth=1.5, label=f'Peak BW ({peak:.0f} GB/s)')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            util_pct = mean / peak * 100
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + peak*0.01,
                    f'{mean:.0f} GB/s\n({util_pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(f'{hw}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=8)
        ax.set_ylabel('Achieved Memory Bandwidth (GB/s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, peak * 1.25)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_bandwidth_utilization.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_bandwidth_by_shape(df):
    """
    Figure 7: Bandwidth utilization by shape for outer-product only.
    Shows how utilization scales with matrix size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        'Outer-Product Bandwidth Utilization by Shape',
        fontsize=13, fontweight='bold'
    )

    for ax, hw in zip(axes, ['T4', 'MI300x']):
        hw_df = df[(df['hardware'] == hw) &
                   (df['algorithm'] == 'outer_product')]
        if hw_df.empty:
            continue

        peak    = PEAK_BW[hw]
        layers  = hw_df['layer'].unique()
        sparsities = sorted(hw_df['sparsity'].unique())
        sp_colors = {0.70: '#1565C0', 0.90: '#42A5F5', 0.95: '#BBDEFB'}

        x     = np.arange(len(layers))
        width = 0.25

        for i, sp in enumerate(sparsities):
            vals = []
            for layer in layers:
                v = hw_df[(hw_df['layer'] == layer) &
                          (hw_df['sparsity'] == sp)]['achieved_bw_GBs'].values
                vals.append(v[0] if len(v) > 0 else 0)
            ax.bar(x + i*width, vals, width,
                   label=f'sparsity={sp:.0%}',
                   color=sp_colors[sp], alpha=0.85)

        ax.axhline(y=peak, color='red', linestyle='--',
                   linewidth=1.5, label=f'Peak ({peak:.0f} GB/s)')
        ax.set_title(f'{hw} — Outer-Product', fontsize=11, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Achieved Bandwidth (GB/s)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig7_outer_product_bandwidth_by_shape.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def print_summary(df):
    """Print key bandwidth statistics."""
    print("=" * 70)
    print("Bandwidth Utilization Summary")
    print("=" * 70)

    for hw in ['T4', 'MI300x']:
        peak = PEAK_BW[hw]
        hw_df = df[df['hardware'] == hw]
        if hw_df.empty:
            continue

        print(f"\n{hw} (peak: {peak} GB/s):")
        print(f"  {'Algorithm':<20} {'Mean BW (GB/s)':>15} {'Utilization':>12} {'Storage':>10}")
        print(f"  {'-'*60}")

        for algo in ALGOS:
            algo_df = hw_df[hw_df['algorithm'] == algo]
            if algo_df.empty:
                continue
            mean_bw   = algo_df['achieved_bw_GBs'].mean()
            util      = mean_bw / peak * 100
            storage   = algo_df['storage_type'].iloc[0]
            print(f"  {algo:<20} {mean_bw:>15.2f} {util:>11.2f}% {storage:>10}")


if __name__ == '__main__':
    df = load_bandwidth_data()
    print_summary(df)
    print("\nGenerating figures...")
    plot_bandwidth_utilization(df)
    plot_bandwidth_by_shape(df)
    print("Done")
