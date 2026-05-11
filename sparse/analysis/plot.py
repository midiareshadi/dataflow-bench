"""
Sparse - Plot Script
Generates all comparison figures for SpMSpM algorithm analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
HW_COLORS  = {'T4': '#FF6F00', 'MI300x': '#6A1B9A'}
ALGOS      = ['vendor', 'inner_product', 'outer_product', 'gustavson']
SPARSITIES = [0.70, 0.90, 0.95]
FONTSIZE   = 11


def load_results():
    t4     = pd.read_csv(os.path.join(RESULTS_DIR, 't4', 't4_results.csv'))
    mi300x = pd.read_csv(os.path.join(RESULTS_DIR, 'mi300x', 'mi300x_results.csv'))
    t4['hardware']     = 'T4'
    mi300x['hardware'] = 'MI300x'
    return pd.concat([t4, mi300x], ignore_index=True)


def plot1_latency_by_algorithm(df):
    """Figure 1: Latency by algorithm per hardware, split by sparsity."""
    layers = df['layer'].unique()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SpMSpM Latency by Algorithm\nLLaMA-7B Attention Shapes',
                 fontsize=13, fontweight='bold')

    x = np.arange(len(layers))
    width = 0.2

    for row, hw in enumerate(['T4', 'MI300x']):
        hw_df = df[df['hardware'] == hw]
        for col, sparsity in enumerate(SPARSITIES):
            ax = axes[row][col]
            sp_df = hw_df[hw_df['sparsity'] == sparsity]
            for i, algo in enumerate(ALGOS):
                vals = [sp_df[(sp_df['layer'] == l) &
                              (sp_df['algorithm'] == algo)]['latency_us'].values[0]
                        for l in layers]
                ax.bar(x + i * width, vals, width,
                       label=algo, color=COLORS[algo], alpha=0.85)

            ax.set_title(f'{hw} — sparsity={sparsity:.0%}',
                         fontsize=10, fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Latency (μs)', fontsize=9)
            ax.set_yscale('log')
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_latency_by_algorithm.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot2_speedup_vs_vendor(df):
    """Figure 2: Speedup vs vendor baseline per algorithm per hardware."""
    layers = df['layer'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Algorithm Speedup vs Vendor Baseline (cuSPARSE/rocSPARSE)',
                 fontsize=13, fontweight='bold')

    algos_no_vendor = ['inner_product', 'outer_product', 'gustavson']
    x = np.arange(len(layers))
    width = 0.25

    for ax, hw in zip(axes, ['T4', 'MI300x']):
        hw_df = df[(df['hardware'] == hw) & (df['sparsity'] == 0.90)]
        for i, algo in enumerate(algos_no_vendor):
            vals = [hw_df[(hw_df['layer'] == l) &
                          (hw_df['algorithm'] == algo)]['speedup_vs_vendor'].values[0]
                    for l in layers]
            ax.bar(x + i * width, vals, width,
                   label=algo, color=COLORS[algo], alpha=0.85)

        ax.axhline(y=1.0, color='black', linestyle='--',
                   linewidth=1.5, label='vendor baseline')
        ax.set_title(f'{hw} (sparsity=90%)', fontsize=FONTSIZE, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Speedup vs Vendor', fontsize=FONTSIZE)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_speedup_vs_vendor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot3_sparsity_sensitivity(df):
    """Figure 3: Speedup vs vendor by sparsity level."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Outer-Product Speedup vs Vendor by Sparsity Level',
                 fontsize=13, fontweight='bold')

    layers = df['layer'].unique()
    x = np.arange(len(layers))
    width = 0.25
    sp_colors = {0.70: '#1565C0', 0.90: '#42A5F5', 0.95: '#BBDEFB'}

    for ax, hw in zip(axes, ['T4', 'MI300x']):
        hw_df = df[(df['hardware'] == hw) &
                   (df['algorithm'] == 'outer_product')]
        for i, sparsity in enumerate(SPARSITIES):
            vals = [hw_df[(hw_df['layer'] == l) &
                          (hw_df['sparsity'] == sparsity)]['speedup_vs_vendor'].values[0]
                    for l in layers]
            ax.bar(x + i * width, vals, width,
                   label=f'sparsity={sparsity:.0%}',
                   color=sp_colors[sparsity], alpha=0.85)

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{hw} — outer_product', fontsize=FONTSIZE, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Speedup vs Vendor', fontsize=FONTSIZE)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_sparsity_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot4_hardware_comparison(df):
    """Figure 4: MI300x speedup over T4 per algorithm."""
    layers = df['layer'].unique()
    fig, ax = plt.subplots(figsize=(14, 5))

    algos_all = ['vendor', 'inner_product', 'outer_product', 'gustavson']
    x = np.arange(len(layers))
    width = 0.2

    sp_df = df[df['sparsity'] == 0.90]
    for i, algo in enumerate(algos_all):
        speedups = []
        for layer in layers:
            t4_lat = sp_df[(sp_df['hardware'] == 'T4') &
                           (sp_df['algorithm'] == algo) &
                           (sp_df['layer'] == layer)]['latency_us'].values[0]
            mi_lat = sp_df[(sp_df['hardware'] == 'MI300x') &
                           (sp_df['algorithm'] == algo) &
                           (sp_df['layer'] == layer)]['latency_us'].values[0]
            speedups.append(t4_lat / mi_lat)
        ax.bar(x + i * width, speedups, width,
               label=algo, color=COLORS[algo], alpha=0.85)

    ax.set_title('MI300x Speedup over T4 per Algorithm (sparsity=90%)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Speedup (T4 / MI300x latency)', fontsize=FONTSIZE)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_hardware_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot5_winner_heatmap(df):
    """Figure 5: Winning algorithm heatmap across hardware, shape, sparsity."""
    algo_map = {'vendor': 0, 'inner_product': 1,
                'outer_product': 2, 'gustavson': 3}
    cmap = matplotlib.colors.ListedColormap(
        [COLORS[a] for a in ['vendor', 'inner_product',
                              'outer_product', 'gustavson']])

    layers     = df['layer'].unique()
    sparsities = sorted(df['sparsity'].unique())
    hardware   = ['T4', 'MI300x']

    # Rows: hardware x sparsity, Cols: layers
    n_rows = len(hardware) * len(sparsities)
    data   = np.zeros((n_rows, len(layers)))
    ylabels = []

    row = 0
    for hw in hardware:
        for sp in sparsities:
            ylabels.append(f'{hw} {sp:.0%}')
            hw_sp_df = df[(df['hardware'] == hw) & (df['sparsity'] == sp)]
            for col, layer in enumerate(layers):
                layer_df = hw_sp_df[hw_sp_df['layer'] == layer]
                best = layer_df.loc[layer_df['latency_us'].idxmin(), 'algorithm']
                data[row, col] = algo_map[best]
            row += 1

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_title('Winning Algorithm Heatmap', fontsize=13, fontweight='bold')

    for i in range(n_rows):
        for j in range(len(layers)):
            algo = ['vendor', 'inner', 'outer', 'gust'][int(data[i, j])]
            ax.text(j, i, algo, ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_winner_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_plots():
    df = load_results()
    print("Generating figures...")
    print("-" * 40)
    plot1_latency_by_algorithm(df)
    plot2_speedup_vs_vendor(df)
    plot3_sparsity_sensitivity(df)
    plot4_hardware_comparison(df)
    plot5_winner_heatmap(df)
    print("-" * 40)
    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    run_plots()
