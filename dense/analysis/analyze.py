"""
Dense Dataflow - Analysis Script
Combines T4 and MI300x results and answers key research questions.
"""

import os
import sys
import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

def load_results():
    t4_path     = os.path.join(RESULTS_DIR, 't4', 't4_results.csv')
    mi300x_path = os.path.join(RESULTS_DIR, 'mi300x', 'mi300x_results.csv')

    t4     = pd.read_csv(t4_path)
    mi300x = pd.read_csv(mi300x_path)

    # Normalize hardware column
    t4['hardware']     = 'T4'
    mi300x['hardware'] = 'MI300x'

    df = pd.concat([t4, mi300x], ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def q1_winning_dataflow_per_hardware(df):
    """Q1: Which dataflow wins on each hardware for each shape?"""
    print("\n" + "="*60)
    print("Q1: Winning dataflow per hardware per shape")
    print("="*60)

    results = []
    for hw in ['T4', 'MI300x']:
        hw_df = df[df['hardware'] == hw]
        for layer in hw_df['layer'].unique():
            layer_df = hw_df[hw_df['layer'] == layer]
            best = layer_df.loc[layer_df['latency_us'].idxmin()]
            results.append({
                'hardware': hw,
                'layer':    layer,
                'winner':   best['dataflow'],
                'latency_us': best['latency_us'],
            })

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    print("\nWinner summary per hardware:")
    for hw in ['T4', 'MI300x']:
        hw_df = result_df[result_df['hardware'] == hw]
        counts = hw_df['winner'].value_counts()
        print(f"  {hw}: {dict(counts)}")

    return result_df


def q2_dataflow_consistency(df):
    """Q2: Is the winning dataflow consistent across operation types?"""
    print("\n" + "="*60)
    print("Q2: Dataflow consistency across operation types")
    print("="*60)

    op_groups = {
        'Attention projections': ['M1_Q_proj', 'M2_K_proj', 'M3_V_proj', 'M4_Out_proj'],
        'FFN layers':            ['M5_FFN_gate', 'M6_FFN_up', 'M7_FFN_down'],
        'Attention scores QKt':  ['M8_AttnScore_512', 'M9_AttnScore_1024', 'M10_AttnScore_2048'],
        'Attention output sV':   ['M11_AttnOut_512', 'M12_AttnOut_1024', 'M13_AttnOut_2048'],
    }

    for hw in ['T4', 'MI300x']:
        print(f"\n  {hw}:")
        hw_df = df[df['hardware'] == hw]
        for group, layers in op_groups.items():
            winners = []
            for layer in layers:
                layer_df = hw_df[hw_df['layer'] == layer]
                if not layer_df.empty:
                    best = layer_df.loc[layer_df['latency_us'].idxmin(), 'dataflow']
                    winners.append(best)
            print(f"    {group}: {winners}")


def q3_os_penalty(df):
    """Q3: How much slower is OS vs best dataflow on each hardware?"""
    print("\n" + "="*60)
    print("Q3: OS penalty vs best dataflow")
    print("="*60)

    for hw in ['T4', 'MI300x']:
        hw_df = df[df['hardware'] == hw]
        penalties = []
        for layer in hw_df['layer'].unique():
            layer_df = hw_df[hw_df['layer'] == layer]
            best_lat = layer_df['latency_us'].min()
            os_lat   = layer_df[layer_df['dataflow'] == 'OS']['latency_us'].values
            if len(os_lat) > 0:
                penalty = os_lat[0] / best_lat
                penalties.append(penalty)

        print(f"\n  {hw} OS penalty vs best:")
        print(f"    Mean:   {np.mean(penalties):.2f}x")
        print(f"    Median: {np.median(penalties):.2f}x")
        print(f"    Max:    {np.max(penalties):.2f}x")
        print(f"    Min:    {np.min(penalties):.2f}x")


def q4_hardware_speedup(df):
    """Q4: How much faster is MI300x vs T4 per dataflow?"""
    print("\n" + "="*60)
    print("Q4: MI300x speedup over T4 per dataflow")
    print("="*60)

    for df_type in ['WS', 'IS', 'OS']:
        t4_df     = df[(df['hardware'] == 'T4')     & (df['dataflow'] == df_type)]
        mi300x_df = df[(df['hardware'] == 'MI300x') & (df['dataflow'] == df_type)]

        merged = t4_df.merge(mi300x_df, on='layer', suffixes=('_t4', '_mi300x'))
        merged['speedup'] = merged['latency_us_t4'] / merged['latency_us_mi300x']

        print(f"\n  {df_type} — MI300x speedup over T4:")
        print(f"    Mean:   {merged['speedup'].mean():.1f}x")
        print(f"    Median: {merged['speedup'].median():.1f}x")
        print(f"    Min:    {merged['speedup'].min():.1f}x")
        print(f"    Max:    {merged['speedup'].max():.1f}x")


def q5_small_vs_large(df):
    """Q5: Does dataflow sensitivity change with matrix size?"""
    print("\n" + "="*60)
    print("Q5: Dataflow sensitivity — small vs large shapes")
    print("="*60)

    small_shapes = ['M8_AttnScore_512', 'M9_AttnScore_1024',
                    'M11_AttnOut_512',  'M12_AttnOut_1024']
    large_shapes = ['M1_Q_proj', 'M2_K_proj', 'M3_V_proj', 'M4_Out_proj',
                    'M5_FFN_gate', 'M6_FFN_up', 'M7_FFN_down']

    for hw in ['T4', 'MI300x']:
        hw_df = df[df['hardware'] == hw]
        print(f"\n  {hw}:")
        for label, shapes in [('Small shapes', small_shapes),
                               ('Large shapes', large_shapes)]:
            subset = hw_df[hw_df['layer'].isin(shapes)]
            spread = []
            for layer in subset['layer'].unique():
                layer_df = subset[subset['layer'] == layer]
                best  = layer_df['latency_us'].min()
                worst = layer_df['latency_us'].max()
                spread.append(worst / best)
            print(f"    {label} — avg worst/best ratio: {np.mean(spread):.2f}x")


def run_analysis():
    df = load_results()
    print(f"Loaded {len(df)} rows total")
    print(f"Hardware: {df['hardware'].unique()}")
    print(f"Dataflows: {df['dataflow'].unique()}")
    print(f"Layers: {len(df['layer'].unique())}")

    q1_winning_dataflow_per_hardware(df)
    q2_dataflow_consistency(df)
    q3_os_penalty(df)
    q4_hardware_speedup(df)
    q5_small_vs_large(df)

    print("\n" + "="*60)
    print("Analysis complete.")
    print("="*60)


if __name__ == '__main__':
    run_analysis()
