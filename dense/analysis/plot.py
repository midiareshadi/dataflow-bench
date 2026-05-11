"""
Dense Dataflow - Plot Script
Generates all comparison figures for T4 vs MI300x dataflow analysis.
Output saved to dense/analysis/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
COLORS = {'WS': '#2196F3', 'IS': '#4CAF50', 'OS': '#F44336'}
HW_COLORS = {'T4': '#FF6F00', 'MI300x': '#6A1B9A'}
FONTSIZE = 11


def load_results():
    t4     = pd.read_csv(os.path.join(RESULTS_DIR, 't4', 't4_results.csv'))
    mi300x = pd.read_csv(os.path.join(RESULTS_DIR, 'mi300x', 'mi300x_results.csv'))
    t4['hardware']     = 'T4'
    mi300x['hardware'] = 'MI300x'
    df = pd.concat([t4, mi300x], ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def plot1_latency_by_dataflow(df):
    """Figure 1: Latency by dataflow — split large vs small shapes per hardware."""
    large_shapes = ['M1_Q_proj', 'M2_K_proj', 'M3_V_proj', 'M4_Out_proj',
                    'M5_FFN_gate', 'M6_FFN_up', 'M7_FFN_down']
    small_shapes = ['M8_AttnScore_512', 'M9_AttnScore_1024', 'M10_AttnScore_2048',
                    'M11_AttnOut_512', 'M12_AttnOut_1024', 'M13_AttnOut_2048']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Latency by Dataflow Strategy\nLLaMA-7B Shapes on T4 and MI300x',
                 fontsize=13, fontweight='bold')

    width = 0.25

    for row, hw in enumerate(['T4', 'MI300x']):
        hw_df = df[df['hardware'] == hw]
        for col, (shapes, label) in enumerate([
            (large_shapes, 'Large Shapes (Projections + FFN)'),
            (small_shapes, 'Small Shapes (Attention Scores)')
        ]):
            ax = axes[row][col]
            x = np.arange(len(shapes))
            for i, df_type in enumerate(['WS', 'IS', 'OS']):
                vals = []
                for l in shapes:
                    row_data = hw_df[(hw_df['layer'] == l) & (hw_df['dataflow'] == df_type)]
                    vals.append(row_data['latency_us'].values[0] if len(row_data) > 0 else 0)
                ax.bar(x + i * width, vals, width, label=df_type,
                       color=COLORS[df_type], alpha=0.85)

            ax.set_title(f'{hw} — {label}', fontsize=10, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(shapes, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Latency (μs)', fontsize=FONTSIZE)
            ax.legend(title='Dataflow', fontsize=9)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f'{x/1000:.1f}ms' if x >= 1000 else f'{x:.0f}μs'))
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_latency_by_dataflow.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot2_winner_heatmap(df):
    """Figure 2: Winning dataflow heatmap — hardware vs layer."""
    layers   = df['layer'].unique()
    hardware = ['T4', 'MI300x']

    winner_map = {'WS': 0, 'IS': 1, 'OS': 2}
    data = np.zeros((len(hardware), len(layers)))

    for i, hw in enumerate(hardware):
        hw_df = df[df['hardware'] == hw]
        for j, layer in enumerate(layers):
            layer_df = hw_df[hw_df['layer'] == layer]
            best = layer_df.loc[layer_df['latency_us'].idxmin(), 'dataflow']
            data[i, j] = winner_map[best]

    fig, ax = plt.subplots(figsize=(14, 3))
    cmap = matplotlib.colors.ListedColormap([COLORS['WS'], COLORS['IS'], COLORS['OS']])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(hardware)))
    ax.set_yticklabels(hardware, fontsize=FONTSIZE)
    ax.set_title('Winning Dataflow per Hardware and Shape', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(len(hardware)):
        for j in range(len(layers)):
            label = ['WS', 'IS', 'OS'][int(data[i, j])]
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_winner_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot3_os_penalty(df):
    """Figure 3: OS penalty vs best dataflow per hardware."""
    fig, ax = plt.subplots(figsize=(14, 5))

    layers = df['layer'].unique()
    x = np.arange(len(layers))
    width = 0.35

    for i, hw in enumerate(['T4', 'MI300x']):
        hw_df = df[df['hardware'] == hw]
        penalties = []
        for layer in layers:
            layer_df = hw_df[hw_df['layer'] == layer]
            best_lat = layer_df['latency_us'].min()
            os_lat   = layer_df[layer_df['dataflow'] == 'OS']['latency_us'].values[0]
            penalties.append(os_lat / best_lat)
        ax.bar(x + i * width, penalties, width, label=hw,
               color=HW_COLORS[hw], alpha=0.85)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Best=1.0')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('OS Latency / Best Latency', fontsize=FONTSIZE)
    ax.set_title('OS Penalty vs Best Dataflow\n(higher = OS is worse relative to best)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_os_penalty.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot4_mi300x_speedup(df):
    """Figure 4: MI300x speedup over T4 per dataflow."""
    fig, ax = plt.subplots(figsize=(14, 5))

    layers = df['layer'].unique()
    x = np.arange(len(layers))
    width = 0.25

    for i, df_type in enumerate(['WS', 'IS', 'OS']):
        speedups = []
        for layer in layers:
            t4_lat = df[(df['hardware'] == 'T4') & (df['dataflow'] == df_type) &
                        (df['layer'] == layer)]['latency_us'].values[0]
            mi_lat = df[(df['hardware'] == 'MI300x') & (df['dataflow'] == df_type) &
                        (df['layer'] == layer)]['latency_us'].values[0]
            speedups.append(t4_lat / mi_lat)
        ax.bar(x + i * width, speedups, width, label=df_type,
               color=COLORS[df_type], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Speedup (T4 latency / MI300x latency)', fontsize=FONTSIZE)
    ax.set_title('MI300x Speedup over T4 per Dataflow Strategy',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Dataflow', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_mi300x_speedup.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot5_tflops_comparison(df):
    """Figure 5: TFLOPS achieved per dataflow per hardware."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Compute Throughput (TFLOPS) by Dataflow Strategy',
                 fontsize=13, fontweight='bold')

    layers = df['layer'].unique()
    x = np.arange(len(layers))
    width = 0.25

    for ax, hw in zip(axes, ['T4', 'MI300x']):
        hw_df = df[df['hardware'] == hw]
        for i, df_type in enumerate(['WS', 'IS', 'OS']):
            vals = [hw_df[(hw_df['layer'] == l) & (hw_df['dataflow'] == df_type)]['TFLOPS'].values[0]
                    for l in layers]
            ax.bar(x + i * width, vals, width, label=df_type,
                   color=COLORS[df_type], alpha=0.85)

        ax.set_title(f'{hw}', fontsize=FONTSIZE, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('TFLOPS', fontsize=FONTSIZE)
        ax.legend(title='Dataflow', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_tflops_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_plots():
    df = load_results()
    print("Generating figures...")
    print("-" * 40)
    plot1_latency_by_dataflow(df)
    plot2_winner_heatmap(df)
    plot3_os_penalty(df)
    plot4_mi300x_speedup(df)
    plot5_tflops_comparison(df)
    print("-" * 40)
    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    run_plots()
