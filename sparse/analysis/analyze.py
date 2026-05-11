"""
Sparse - Analysis Script
Combines T4 and MI300x SpMSpM results and answers key research questions.
"""

import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')


def load_results():
    t4_path     = os.path.join(RESULTS_DIR, 't4', 't4_results.csv')
    mi300x_path = os.path.join(RESULTS_DIR, 'mi300x', 'mi300x_results.csv')

    t4     = pd.read_csv(t4_path)
    mi300x = pd.read_csv(mi300x_path)

    t4['hardware']     = 'T4'
    mi300x['hardware'] = 'MI300x'

    df = pd.concat([t4, mi300x], ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def q1_winning_algorithm(df):
    """Q1: Which algorithm wins on each hardware for each shape/sparsity?"""
    print("\n" + "=" * 60)
    print("Q1: Winning algorithm per hardware")
    print("=" * 60)

    for hw in ['T4', 'MI300x']:
        hw_df = df[df['hardware'] == hw]
        print(f"\n{hw}:")
        wins = {}
        for (layer, sparsity), group in hw_df.groupby(['layer', 'sparsity']):
            best = group.loc[group['latency_us'].idxmin(), 'algorithm']
            wins[best] = wins.get(best, 0) + 1
        print(f"  Winner counts: {wins}")

        # Show outer-product speedup stats
        op_df = hw_df[hw_df['algorithm'] == 'outer_product']
        print(f"  outer_product speedup vs vendor: "
              f"mean={op_df['speedup_vs_vendor'].mean():.1f}x "
              f"max={op_df['speedup_vs_vendor'].max():.1f}x "
              f"min={op_df['speedup_vs_vendor'].min():.1f}x")


def q2_algorithm_vs_vendor(df):
    """Q2: Do any algorithms beat vendor baseline?"""
    print("\n" + "=" * 60)
    print("Q2: Algorithm speedup vs vendor baseline")
    print("=" * 60)

    algos = ['inner_product', 'outer_product', 'gustavson']
    for hw in ['T4', 'MI300x']:
        hw_df = df[df['hardware'] == hw]
        print(f"\n{hw}:")
        for algo in algos:
            algo_df = hw_df[hw_df['algorithm'] == algo]
            beats = (algo_df['speedup_vs_vendor'] > 1.0).sum()
            total = len(algo_df)
            mean_speedup = algo_df['speedup_vs_vendor'].mean()
            print(f"  {algo:<15}: beats vendor {beats}/{total} cases | "
                  f"mean speedup {mean_speedup:.1f}x")


def q3_sparsity_sensitivity(df):
    """Q3: How does sparsity level affect algorithm ranking?"""
    print("\n" + "=" * 60)
    print("Q3: Sparsity sensitivity")
    print("=" * 60)

    for hw in ['T4', 'MI300x']:
        hw_df = df[df['hardware'] == hw]
        print(f"\n{hw} — outer_product speedup vs vendor by sparsity:")
        op_df = hw_df[hw_df['algorithm'] == 'outer_product']
        for sparsity in sorted(df['sparsity'].unique()):
            sp_df = op_df[op_df['sparsity'] == sparsity]
            print(f"  sparsity={sparsity:.0%}: "
                  f"mean={sp_df['speedup_vs_vendor'].mean():.1f}x "
                  f"max={sp_df['speedup_vs_vendor'].max():.1f}x")


def q4_hardware_comparison(df):
    """Q4: How do results differ between T4 and MI300x?"""
    print("\n" + "=" * 60)
    print("Q4: T4 vs MI300x comparison")
    print("=" * 60)

    algos = ['vendor', 'inner_product', 'outer_product', 'gustavson']
    for algo in algos:
        t4_df     = df[(df['hardware'] == 'T4')     & (df['algorithm'] == algo)]
        mi300x_df = df[(df['hardware'] == 'MI300x') & (df['algorithm'] == algo)]
        merged = t4_df.merge(mi300x_df, on=['layer', 'sparsity'],
                             suffixes=('_t4', '_mi300x'))
        merged['speedup'] = merged['latency_us_t4'] / merged['latency_us_mi300x']
        print(f"  {algo:<15}: MI300x speedup over T4 — "
              f"mean={merged['speedup'].mean():.1f}x "
              f"max={merged['speedup'].max():.1f}x "
              f"min={merged['speedup'].min():.1f}x")


def q5_inner_vs_gustavson(df):
    """Q5: How do inner-product and Gustavson compare?"""
    print("\n" + "=" * 60)
    print("Q5: Inner-product vs Gustavson comparison")
    print("=" * 60)

    for hw in ['T4', 'MI300x']:
        hw_df = df[df['hardware'] == hw]
        inner = hw_df[hw_df['algorithm'] == 'inner_product']
        gust  = hw_df[hw_df['algorithm'] == 'gustavson']
        merged = inner.merge(gust, on=['layer', 'sparsity'],
                             suffixes=('_inner', '_gust'))
        merged['ratio'] = merged['latency_us_inner'] / merged['latency_us_gust']
        gust_wins = (merged['ratio'] > 1.0).sum()
        inner_wins = (merged['ratio'] < 1.0).sum()
        print(f"  {hw}: Gustavson wins {gust_wins}/18, "
              f"Inner wins {inner_wins}/18 | "
              f"mean ratio inner/gust={merged['ratio'].mean():.2f}")


def run_analysis():
    df = load_results()
    print(f"Loaded {len(df)} rows total")
    print(f"Hardware: {df['hardware'].unique().tolist()}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")
    print(f"Shapes: {df['layer'].nunique()}")
    print(f"Sparsity levels: {sorted(df['sparsity'].unique().tolist())}")

    q1_winning_algorithm(df)
    q2_algorithm_vs_vendor(df)
    q3_sparsity_sensitivity(df)
    q4_hardware_comparison(df)
    q5_inner_vs_gustavson(df)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == '__main__':
    run_analysis()
