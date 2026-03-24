"""
Comparison script to train and evaluate both RF and Conv1D models.

Usage:
python results.py \
  --features openmic-2018.npz \
  --labels openmic-2018-aggregated-labels.csv
"""

import argparse
import os
import subprocess    
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import numpy as np


def run_model(model_type, features_path, labels_path, output_prefix, extra_args=None):
    """Run a model and capture results."""
    print(f"\n{'='*60}")
    print(f"Running {model_type.upper()} model...")
    print(f"{'='*60}\n")
    
    cmd = [
        'python', 'openmic_temporal_conv1d.py',
        '--features', features_path,
        '--labels', labels_path,
        '--model-type', model_type,
        '--out', f'{output_prefix}_{model_type}.{"joblib" if model_type == "rf" else "h5"}'
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Parse metrics from output
    metrics = {
        'model': model_type,
        'training_time_seconds': elapsed_time
    }
    
    for line in result.stdout.split('\n'):
        if 'Micro F1' in line:
            metrics['micro_f1'] = float(line.split(':')[1].strip())
        elif 'Macro F1' in line:
            metrics['macro_f1'] = float(line.split(':')[1].strip())
        elif 'Macro AUC' in line and 'None' not in line:
            try:
                metrics['macro_auc'] = float(line.split(':')[1].strip())
            except:
                metrics['macro_auc'] = None
    
    return metrics


def plot_comparison(comparison_df, output_dir='.'):
    """
    Plot comparison results as bar charts and a combined metrics line plot.
    
    Args:
        comparison_df: DataFrame with columns model, micro_f1, macro_f1, macro_auc, training_time_seconds
        output_dir: Directory to save plot images
    """
    models = comparison_df['model'].str.upper().tolist()
    x = np.arange(len(models))
    width = 0.35

    # Prepare metric columns (handle NaN/None)
    metrics_to_plot = ['micro_f1', 'macro_f1', 'macro_auc']
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    for col in available_metrics:
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce').fillna(0)

    # ---- 1. Grouped Bar Chart: F1 and AUC metrics ----
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_metrics = len(available_metrics)
    bar_width = 0.25
    offsets = np.linspace(-bar_width * (n_metrics - 1) / 2, bar_width * (n_metrics - 1) / 2, n_metrics)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    metric_labels = {'micro_f1': 'Micro F1', 'macro_f1': 'Macro F1', 'macro_auc': 'Macro AUC'}
    
    for i, metric in enumerate(available_metrics):
        values = comparison_df[metric].tolist()
        bars = ax.bar(x + offsets[i], values, bar_width, label=metric_labels.get(metric, metric), color=colors[i % len(colors)])
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Comparison: F1 and AUC Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = f'{output_dir}/model_comparison_metrics.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar chart to: {bar_path}")
    plt.close()

    # ---- 2. Bar Chart: Training Time ----
    fig, ax = plt.subplots(figsize=(8, 5))
    times = comparison_df['training_time_seconds'].tolist()
    colors_time = ['#2E86AB', '#E63946']
    bars = ax.bar(models, times, color=colors_time[:len(models)])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Comparison: Training Time', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    time_path = f'{output_dir}/model_comparison_training_time.png'
    plt.savefig(time_path, dpi=150, bbox_inches='tight')
    print(f"Saved training time chart to: {time_path}")
    plt.close()

    # ---- 3. Line Plot: Metrics by Model ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x_line = np.arange(len(models))
    
    for i, metric in enumerate(available_metrics):
        values = comparison_df[metric].tolist()
        ax.plot(x_line, values, marker='o', linewidth=2, markersize=10, 
                label=metric_labels.get(metric, metric), color=colors[i % len(colors)])
        for j, v in enumerate(values):
            ax.annotate(f'{v:.3f}', (x_line[j], v), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    ax.set_xticks(x_line)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Comparison: Metrics Across Models (Line Plot)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    line_path = f'{output_dir}/model_comparison_line.png'
    plt.savefig(line_path, dpi=150, bbox_inches='tight')
    print(f"Saved line plot to: {line_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare RF and Conv1D models')
    parser.add_argument('--features', required=True, help='Path to .npz features')
    parser.add_argument('--labels', required=True, help='Path to labels CSV')
    parser.add_argument('--output-prefix', default='model', help='Prefix for output files')
    parser.add_argument('--output-dir', default='.', help='Directory to save comparison CSV and plots')
    
    # Optional model-specific overrides
    parser.add_argument('--rf-estimators', type=int, default=200, 
                       help='Number of trees for RF')
    parser.add_argument('--conv1d-epochs', type=int, default=50,
                       help='Epochs for Conv1D')
    parser.add_argument('--conv1d-batch-size', type=int, default=32,
                       help='Batch size for Conv1D')
    
    args = parser.parse_args()
    
    results = []
    
    # Run RandomForest
    rf_args = ['--n_estimators', str(args.rf_estimators)]
    rf_metrics = run_model('rf', args.features, args.labels, args.output_prefix, rf_args)
    results.append(rf_metrics)
    
    # Run Conv1D
    conv_args = [
        '--epochs', str(args.conv1d_epochs),
        '--batch_size', str(args.conv1d_batch_size)
    ]
    conv_metrics = run_model('conv1d', args.features, args.labels, args.output_prefix, conv_args)
    results.append(conv_metrics)
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60 + "\n")
    
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = f'{args.output_dir}/model_comparison.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nComparison saved to: {csv_path}")
    
    # Plot bar and line charts
    print("\nGenerating comparison plots...")
    plot_comparison(comparison_df, output_dir=args.output_dir)
    
    # Determine winner
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")
    
    if 'macro_f1' in comparison_df.columns:
        best_idx = comparison_df['macro_f1'].idxmax()
        best_model = comparison_df.loc[best_idx, 'model']
        best_f1 = comparison_df.loc[best_idx, 'macro_f1']
        
        print(f"Best Macro F1: {best_model.upper()} ({best_f1:.4f})")
        
        rf_f1 = comparison_df[comparison_df['model'] == 'rf']['macro_f1'].values[0]
        conv_f1 = comparison_df[comparison_df['model'] == 'conv1d']['macro_f1'].values[0]
        improvement = ((conv_f1 - rf_f1) / rf_f1) * 100
        
        print(f"Conv1D vs RF improvement: {improvement:+.2f}%")
    
    print(f"\nTraining times:")
    for _, row in comparison_df.iterrows():
        print(f"  {row['model'].upper()}: {row['training_time_seconds']:.1f}s")


if __name__ == '__main__':
    main()