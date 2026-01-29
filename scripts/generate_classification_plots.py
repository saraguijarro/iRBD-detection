#!/usr/bin/env python3
"""
Generate Classification Performance Visualizations
This script creates comprehensive plots for the thesis Results chapter,
extracting data from LSTM and ML baseline log files.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import roc_curve, auc

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("pastel")

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_DIR = Path("/work3/s184484/iRBD-detection")
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = PROJECT_DIR / "results" / "classification_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Font sizes for consistency
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10

# All preprocessing versions
VERSIONS = ['v0', 'v1', 'v1t', 'vvt']

# Color scheme for versions (consistent across all plots)
VERSION_COLORS = {
    'v0': '#E57373',   # Red - Baseline
    'v1': '#64B5F6',   # Blue - 18°C rate-of-change
    'v1t': '#81C784',  # Green - 20°C rate-of-change
    'vvt': '#FFB74D'   # Orange - 20°C night-level
}

# Labels for versions (used in legends)
VERSION_LABELS = {
    'v0': 'v0',
    'v1': 'v1',
    'v1t': 'v1t',
    'vvt': 'vvt'
}

# ML model names
ML_MODELS = ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost']


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_lstm_results(log_file):
    """Extract LSTM results from log file."""
    if not log_file.exists():
        print(f"   WARNING: LSTM log file not found: {log_file}")
        return {'participant_metrics': {}, 'fold_metrics': [], 'confusion_matrix': None}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    results = {
        'participant_metrics': {},
        'fold_metrics': []
    }
    
    # Extract average participant-level metrics from summary
    avg_pattern = r'Participant-level \(CLINICAL PERFORMANCE\):.*?Accuracy:\s+([\d.]+)\s+±.*?Precision:\s+([\d.]+)\s+±.*?Recall:\s+([\d.]+)\s+±.*?F1:\s+([\d.]+)\s+±'
    avg_match = re.search(avg_pattern, content, re.DOTALL)
    if avg_match:
        results['participant_metrics'] = {
            'accuracy': float(avg_match.group(1)),
            'precision': float(avg_match.group(2)),
            'recall': float(avg_match.group(3)),
            'f1': float(avg_match.group(4))
        }
    
    # Extract per-fold metrics
    fold_sections = re.findall(
        r'FOLD (\d+)/5.*?Participant-level \(CLINICAL\):.*?Accuracy: ([\d.]+).*?Precision: ([\d.]+).*?Recall: ([\d.]+).*?F1: ([\d.]+)', 
        content, re.DOTALL
    )
    for fold_num, acc, prec, rec, f1 in fold_sections:
        results['fold_metrics'].append({
            'fold': int(fold_num),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        })
    
    # Estimate confusion matrix from average metrics
    if results['participant_metrics']:
        total_participants = 84
        test_size_per_fold = total_participants // 5
        irbd_per_fold = test_size_per_fold // 2
        control_per_fold = test_size_per_fold // 2
        
        recall = results['participant_metrics']['recall']
        precision = results['participant_metrics']['precision']
        
        tp = int(recall * irbd_per_fold * 5)
        fn = int((1 - recall) * irbd_per_fold * 5)
        
        if precision > 0:
            fp = int(tp / precision - tp)
        else:
            fp = 0
        
        tn = control_per_fold * 5 - fp
        results['confusion_matrix'] = np.array([[tn, fp], [fn, tp]])
    
    return results


def extract_ml_results(log_file):
    """Extract ML baseline results from log file."""
    if not log_file.exists():
        print(f"   WARNING: ML log file not found: {log_file}")
        return {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    results = {}
    
    for model in ML_MODELS:
        pattern = rf'{model} - Average Results:.*?Participant-level \(CLINICAL PERFORMANCE\):.*?Accuracy:\s+([\d.]+)\s+±.*?Precision:\s+([\d.]+)\s+±.*?Recall:\s+([\d.]+)\s+±'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            acc = float(match.group(1))
            prec = float(match.group(2))
            rec = float(match.group(3))
            if prec + rec > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
            else:
                f1 = 0.0
            
            results[model] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
    
    return results



def load_json_results(version, model_type='lstm'):
    """Load results from JSON file for a specific version and model type."""
    if model_type == 'lstm':
        results_dir = RESULTS_DIR / f"lstm_{version}_night_level"
    else:
        results_dir = RESULTS_DIR / f"ml_baselines_{version}"
    
    if not results_dir.exists():
        print(f"   WARNING: Results directory not found: {results_dir}")
        return None
    
    json_files = sorted(results_dir.glob("results_*.json"), reverse=True)
    if not json_files:
        print(f"   WARNING: No JSON results found in {results_dir}")
        return None
    
    latest_file = json_files[0]
    print(f"   Loading JSON: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def extract_confusion_matrix_from_json(json_data):
    """
    Extract the actual confusion matrix from JSON results by aggregating
    predictions across all folds.
    
    Args:
        json_data: Loaded JSON results dictionary
    
    Returns:
        numpy array of shape (2, 2) representing the confusion matrix
    """
    if json_data is None:
        return None
    
    all_y_true = []
    all_y_pred = []
    
    for fold in json_data.get('fold_results', []):
        pr = fold.get('participant_results', {})
        y_true = pr.get('y_true', [])
        y_pred_prob = pr.get('y_pred_prob', [])
        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    
    if not all_y_true:
        return None
    
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])


def load_feature_importance(version, model='rf'):
    """
    Load feature importance from saved file.
    
    Args:
        version: Preprocessing version (v0, v1, v1t, vvt)
        model: 'rf' for Random Forest or 'xgb' for XGBoost
    """
    results_dir = RESULTS_DIR / f"ml_baselines_{version}"
    importance_file = results_dir / f"feature_importance_{model}.json"
    
    if not importance_file.exists():
        print(f"   WARNING: Feature importance not found: {importance_file}")
        return None
    
    with open(importance_file, 'r') as f:
        return json.load(f)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_model_comparison(lstm_results, ml_results):
    """
    Create comprehensive model performance comparison across all versions.
    Shows LSTM + 4 ML models for each metric, grouped by preprocessing version.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Model Performance Comparison - iRBD Classification\nAcross Preprocessing Versions', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score']
    all_models = ['LSTM'] + ML_MODELS
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(all_models))
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        
        for i, version in enumerate(VERSIONS):
            values = []
            for model in all_models:
                if model == 'LSTM':
                    val = lstm_results[version].get('participant_metrics', {}).get(metric, 0) * 100
                else:
                    val = ml_results[version].get(model, {}).get(metric, 0) * 100
                values.append(val)
            
            bars = ax.bar(x + offsets[i] * width, values, width, 
                         label=VERSION_LABELS[version], 
                         color=VERSION_COLORS[version], alpha=0.85)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=7, rotation=90)
        
        ax.set_ylabel(f'{metric_name} (%)', fontsize=LABEL_SIZE)
        ax.set_title(metric_name, fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=TICK_SIZE)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([50, 105])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'model_performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(lstm_results, json_results=None):
    """
    Create confusion matrices for LSTM across all four versions.
    2x2 grid showing confusion matrices with counts and percentages.
    
    If json_results is provided, uses actual confusion matrices from JSON files.
    Otherwise falls back to estimated matrices from log files (less accurate).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('LSTM Confusion Matrices - Participant-Level Classification\nAcross Preprocessing Versions', 
                 fontsize=16, fontweight='bold')
    
    for idx, version in enumerate(VERSIONS):
        ax = axes[idx // 2, idx % 2]
        
        # Prefer JSON results (actual data) over log file estimates
        if json_results is not None and version in json_results:
            cm = extract_confusion_matrix_from_json(json_results[version])
        else:
            cm = lstm_results[version].get('confusion_matrix')
        
        if cm is None:
            ax.text(0.5, 0.5, f'No data for {version}', ha='center', va='center', 
                   fontsize=14, transform=ax.transAxes)
            ax.set_title(VERSION_LABELS[version], fontsize=TITLE_SIZE, fontweight='bold')
            ax.axis('off')
            continue
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=['Control', 'iRBD'], yticklabels=['Control', 'iRBD'])
        
        # Add custom annotations with counts and percentages
        for i in range(2):
            for j in range(2):
                text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       color='white' if cm[i, j] > cm.max() / 2 else 'black',
                       fontsize=12, fontweight='bold')
        
        ax.set_title(VERSION_LABELS[version], fontsize=TITLE_SIZE, fontweight='bold', pad=10)
        ax.set_ylabel('True Label', fontsize=LABEL_SIZE)
        ax.set_xlabel('Predicted Label', fontsize=LABEL_SIZE)
        
        # Calculate and display metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        
        metrics_text = f'Acc: {accuracy:.1f}%\nSens: {sensitivity:.1f}%\nSpec: {specificity:.1f}%'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=VERSION_COLORS[version], alpha=0.3))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fold_variability(lstm_results):
    """
    Create box plots showing per-fold performance variability across all versions.
    Shows distribution of metrics across 5-fold CV for each preprocessing version.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LSTM Per-Fold Performance Variability\nAcross Preprocessing Versions', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Collect data for all versions
        data = []
        labels = []
        colors = []
        
        for version in VERSIONS:
            fold_metrics = lstm_results[version].get('fold_metrics', [])
            if fold_metrics:
                values = [fold[metric] * 100 for fold in fold_metrics]
                data.append(values)
                labels.append(VERSION_LABELS[version].replace(' ', '\n'))
                colors.append(VERSION_COLORS[version])
        
        if not data:
            ax.text(0.5, 0.5, 'No fold data available', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_title(metric_name, fontsize=TITLE_SIZE, fontweight='bold')
            continue
        
        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
        
        # Color median line
        for median in bp['medians']:
            median.set_color('grey')
            median.set_linewidth(2.5)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points with jitter
        np.random.seed(42)  # For reproducibility
        for i, (values, color) in enumerate(zip(data, colors), 1):
            x = np.random.normal(i, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.6, s=50, color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(f'{metric_name} (%)', fontsize=LABEL_SIZE)
        ax.set_title(metric_name, fontsize=TITLE_SIZE, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([50, 105])
        
        # Add mean annotations
        for i, values in enumerate(data, 1):
            mean_val = np.mean(values)
            ax.hlines(mean_val, i - 0.3, i + 0.3, colors='black', linestyles='--', linewidth=2)
            ax.text(i + 0.35, mean_val, f'μ={mean_val:.1f}%', va='center', fontsize=8, fontweight='bold')
        
        # Add statistics text box
        stats_lines = []
        for version, values in zip(VERSIONS, data):
            if values:
                stats_lines.append(f'{version}: {np.mean(values):.1f}% ± {np.std(values):.1f}%')
        stats_text = '\n'.join(stats_lines)
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='#FFE082', alpha=0.5))
        
        # Add legend (only on first subplot to avoid repetition)
        if idx == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Line2D([0], [0], color='grey', linewidth=2.5, label='Median'),
                Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Mean (μ)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', 
                    markeredgecolor='black', markersize=8, label='Individual fold values')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fold_variability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_preprocessing_impact(lstm_results):
    """
    Visualize the impact of different preprocessing strategies on LSTM performance.
    Left: All metrics by version. Right: Change relative to v0 baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Impact of Preprocessing Strategies on LSTM Performance', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Left plot: All metrics across versions
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    
    for i, version in enumerate(VERSIONS):
        values = []
        for metric in metrics:
            val = lstm_results[version].get('participant_metrics', {}).get(metric, 0) * 100
            values.append(val)
        
        bars = ax1.bar(x + offsets[i] * width, values, width,
                      label=VERSION_LABELS[version],
                      color=VERSION_COLORS[version], alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Performance (%)', fontsize=LABEL_SIZE)
    ax1.set_title('Performance Metrics by Preprocessing Version', fontsize=TITLE_SIZE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, fontsize=TICK_SIZE)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([50, 105])
    
    # Right plot: Improvement relative to baseline (v0)
    ax2 = axes[1]
    baseline_metrics = lstm_results['v0'].get('participant_metrics', {})
    
    if baseline_metrics:
        comparison_versions = ['v1', 'v1t', 'vvt']
        x = np.arange(len(metrics))
        width = 0.25
        offsets = [-1, 0, 1]
        
        for i, version in enumerate(comparison_versions):
            improvements = []
            for metric in metrics:
                baseline_val = baseline_metrics.get(metric, 0) * 100
                version_val = lstm_results[version].get('participant_metrics', {}).get(metric, 0) * 100
                improvements.append(version_val - baseline_val)
            
            bars = ax2.bar(x + offsets[i] * width, improvements, width,
                          label=VERSION_LABELS[version],
                          color=VERSION_COLORS[version], alpha=0.85)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                label = f'+{height:.1f}' if height >= 0 else f'{height:.1f}'
                va = 'bottom' if height >= 0 else 'top'
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va=va, fontsize=8, fontweight='bold')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Change from Baseline (%)', fontsize=LABEL_SIZE)
        ax2.set_title('Performance Change Relative to v0', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names, fontsize=TICK_SIZE)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No baseline (v0) data available', ha='center', va='center',
                fontsize=14, transform=ax2.transAxes)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'preprocessing_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_best_model_summary(lstm_results, ml_results):
    """
    Create a heatmap showing F1-scores for all model×version combinations.
    Highlights the best model for each preprocessing version.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Model Performance Summary by Preprocessing Version', 
                 fontsize=16, fontweight='bold')
    
    all_models = ['LSTM'] + ML_MODELS
    
    # Create data for heatmap
    f1_scores = np.zeros((len(VERSIONS), len(all_models)))
    
    for i, version in enumerate(VERSIONS):
        for j, model in enumerate(all_models):
            if model == 'LSTM':
                f1 = lstm_results[version].get('participant_metrics', {}).get('f1', 0) * 100
            else:
                f1 = ml_results[version].get(model, {}).get('f1', 0) * 100
            f1_scores[i, j] = f1
    
    # Create heatmap
    sns.heatmap(f1_scores, annot=True, fmt='.1f', cmap='RdYlGn',
               xticklabels=all_models, yticklabels=[VERSION_LABELS[v] for v in VERSIONS],
               ax=ax, cbar_kws={'label': 'F1-Score (%)'}, vmin=50, vmax=100)
    
    ax.set_title('F1-Score (%) by Model and Preprocessing Version', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('Model', fontsize=LABEL_SIZE)
    ax.set_ylabel('Preprocessing Version', fontsize=LABEL_SIZE)
    
    # Highlight best model per version with black border
    for i in range(len(VERSIONS)):
        best_idx = np.argmax(f1_scores[i, :])
        ax.add_patch(plt.Rectangle((best_idx, i), 1, 1, fill=False, edgecolor='black', linewidth=3))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'best_model_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_roc_curves_by_version(json_results):
    """Plot ROC curves comparing preprocessing versions for LSTM."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for version in VERSIONS:
        if version not in json_results or json_results[version] is None:
            continue
        
        all_y_true = []
        all_y_pred_prob = []
        
        for fold_result in json_results[version].get('fold_results', []):
            participant_results = fold_result.get('participant_results', {})
            all_y_true.extend(participant_results.get('y_true', []))
            all_y_pred_prob.extend(participant_results.get('y_pred_prob', []))
        
        if len(all_y_true) == 0:
            continue
        
        fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=VERSION_COLORS[version], linewidth=2,
                label=f'{VERSION_LABELS[version]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=LABEL_SIZE)
    ax.set_title('ROC Curves: LSTM by Preprocessing Version', fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'roc_curves_by_version.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_vs_nights(json_results, version='vvt'):
    """Plot performance as a function of number of nights used."""
    if version not in json_results or json_results[version] is None:
        print(f"   Skipping performance vs nights (no data for {version})")
        return
    
    # Collect all participant data across folds
    all_participants = {}
    for fold_result in json_results[version].get('fold_results', []):
        participant_results = fold_result.get('participant_results', {})
        pids = participant_results.get('participant_ids', [])
        labels = participant_results.get('y_true', [])
        night_probs_list = participant_results.get('night_probs', [])
        
        for pid, label, probs in zip(pids, labels, night_probs_list):
            if pid not in all_participants:
                all_participants[pid] = {'y_true': label, 'night_probs': probs}
    
    if not all_participants:
        print(f"   Skipping performance vs nights (no participant data)")
        return
    
    max_nights = min(max(len(p['night_probs']) for p in all_participants.values()), 14)
    accuracies, sensitivities, specificities = [], [], []
    
    for n_nights in range(1, max_nights + 1):
        y_true_list, y_pred_list = [], []
        for pid, data in all_participants.items():
            if len(data['night_probs']) >= n_nights:
                avg_prob = np.mean(data['night_probs'][:n_nights])
                y_true_list.append(data['y_true'])
                y_pred_list.append(1 if avg_prob >= 0.5 else 0)
        
        if len(y_true_list) == 0:
            continue
            
        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        
        accuracies.append(np.mean(y_true_arr == y_pred_arr))
        tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
        fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
        tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
        fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    nights_range = range(1, len(accuracies) + 1)
    ax.plot(list(nights_range), accuracies, 'o-', color='#1565C0', linewidth=2, markersize=8, label='Accuracy')
    ax.plot(list(nights_range), sensitivities, 's-', color='#2E7D32', linewidth=2, markersize=8, label='Sensitivity')
    ax.plot(list(nights_range), specificities, '^-', color='#F57C00', linewidth=2, markersize=8, label='Specificity')
    
    ax.set_xlabel('Number of Nights', fontsize=LABEL_SIZE)
    ax.set_ylabel('Score', fontsize=LABEL_SIZE)
    ax.set_title(f'Classification Performance vs Number of Nights ({version})', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlim([0.5, len(accuracies) + 0.5])
    ax.set_ylim([0.5, 1.02])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'performance_vs_nights_{version}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_2d_prediction_scatter(json_results, version1='v0', version2='vvt'):
    """Plot 2D scatter comparing predictions from two preprocessing versions."""
    if version1 not in json_results or version2 not in json_results:
        print(f"   Skipping 2D scatter (missing data for {version1} or {version2})")
        return
    if json_results[version1] is None or json_results[version2] is None:
        print(f"   Skipping 2D scatter (no data for {version1} or {version2})")
        return
    
    # Collect predictions from version1
    v1_predictions = {}
    for fold_result in json_results[version1].get('fold_results', []):
        pr = fold_result.get('participant_results', {})
        for pid, prob, label in zip(pr.get('participant_ids', []), 
                                     pr.get('y_pred_prob', []), 
                                     pr.get('y_true', [])):
            v1_predictions[pid] = {'prob': prob, 'label': label}
    
    # Collect predictions from version2
    v2_predictions = {}
    for fold_result in json_results[version2].get('fold_results', []):
        pr = fold_result.get('participant_results', {})
        for pid, prob, label in zip(pr.get('participant_ids', []), 
                                     pr.get('y_pred_prob', []), 
                                     pr.get('y_true', [])):
            v2_predictions[pid] = {'prob': prob, 'label': label}
    
    # Find common participants
    common_pids = set(v1_predictions.keys()) & set(v2_predictions.keys())
    if len(common_pids) == 0:
        print(f"   Skipping 2D scatter (no common participants)")
        return
    
    # Prepare data
    x_probs, y_probs, labels = [], [], []
    for pid in common_pids:
        x_probs.append(v1_predictions[pid]['prob'])
        y_probs.append(v2_predictions[pid]['prob'])
        labels.append(v1_predictions[pid]['label'])
    
    x_probs = np.array(x_probs)
    y_probs = np.array(y_probs)
    labels = np.array(labels)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot controls
    control_mask = labels == 0
    ax.scatter(x_probs[control_mask], y_probs[control_mask], 
               c='#1565C0', s=80, alpha=0.7, label='Control', edgecolors='white', linewidth=0.5)
    
    # Plot iRBD
    irbd_mask = labels == 1
    ax.scatter(x_probs[irbd_mask], y_probs[irbd_mask], 
               c='#C62828', s=80, alpha=0.7, label='iRBD', edgecolors='white', linewidth=0.5)
    
    # Add decision boundaries
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add diagonal
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(f'{VERSION_LABELS[version1]} Prediction Probability', fontsize=LABEL_SIZE)
    ax.set_ylabel(f'{VERSION_LABELS[version2]} Prediction Probability', fontsize=LABEL_SIZE)
    ax.set_title(f'Prediction Comparison: {VERSION_LABELS[version1]} vs {VERSION_LABELS[version2]}', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'prediction_scatter_{version1}_vs_{version2}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_importance(version='vvt', top_n=20):
    """Plot feature importance from Random Forest and XGBoost side by side."""
    rf_data = load_feature_importance(version, 'rf')
    xgb_data = load_feature_importance(version, 'xgb')
    
    if rf_data is None and xgb_data is None:
        print(f"   Skipping feature importance plot (no data for {version})")
        return
    
    n_plots = sum([rf_data is not None, xgb_data is not None])
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 10))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if rf_data is not None:
        ax = axes[plot_idx]
        importances = np.array(rf_data['importances'])
        feature_names = rf_data['feature_names']
        
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[sorted_indices]
        top_names = [feature_names[i] for i in sorted_indices]
        
        y_pos = np.arange(len(top_importances))
        ax.barh(y_pos, top_importances, color='#2E7D32', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=LABEL_SIZE)
        ax.set_title(f'Random Forest\nTop {top_n} Features ({version})', 
                     fontsize=TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plot_idx += 1
    
    if xgb_data is not None:
        ax = axes[plot_idx]
        importances = np.array(xgb_data['importances'])
        feature_names = xgb_data['feature_names']
        
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[sorted_indices]
        top_names = [feature_names[i] for i in sorted_indices]
        
        y_pos = np.arange(len(top_importances))
        ax.barh(y_pos, top_importances, color='#C62828', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=LABEL_SIZE)
        ax.set_title(f'XGBoost\nTop {top_n} Features ({version})', 
                     fontsize=TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'feature_importance_{version}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to generate all plots."""
    print("=" * 70)
    print("Generating Classification Performance Visualizations")
    print("All Preprocessing Versions: v0, v1, v1t, vvt")
    print("=" * 70)
    
    # PART 1: Load data from log files (for existing plots)
    print("\n1. Loading data from log files...")
    
    lstm_results = {}
    ml_results = {}
    
    for version in VERSIONS:
        lstm_log = RESULTS_DIR / f"lstm_{version}_night_level" / "training.log"
        ml_log = RESULTS_DIR / f"ml_baselines_{version}" / "training.log"
        
        print(f"\n   Loading {version}...")
        lstm_results[version] = extract_lstm_results(lstm_log)
        ml_results[version] = extract_ml_results(ml_log)
    
    # PART 2: Load data from JSON files (for new plots)
    print("\n2. Loading data from JSON result files...")
    
    json_results_lstm = {}
    for version in VERSIONS:
        print(f"   Loading JSON for {version}...")
        json_results_lstm[version] = load_json_results(version, 'lstm')
    
    # PART 3: Generate plots
    print("\n3. Generating plots...")
    print("-" * 40)
    
    plot_model_comparison(lstm_results, ml_results)
    plot_confusion_matrices(lstm_results, json_results_lstm)  # Pass JSON results for accurate confusion matrices
    plot_fold_variability(lstm_results)
    plot_preprocessing_impact(lstm_results)
    plot_best_model_summary(lstm_results, ml_results)
    
    # PART 4: Generate more plots
    print("\n4. Generating more plots...")
    print("-" * 40)
    
    plot_roc_curves_by_version(json_results_lstm)
    plot_performance_vs_nights(json_results_lstm, version='vvt')
    
    # 2D Prediction Scatter plots (v0 vs each other version)
    plot_2d_prediction_scatter(json_results_lstm, version1='v0', version2='v1')
    plot_2d_prediction_scatter(json_results_lstm, version1='v0', version2='v1t')
    plot_2d_prediction_scatter(json_results_lstm, version1='v0', version2='vvt')
    
    plot_feature_importance(version='vvt')
    
    print("\n" + "=" * 70)
    print("All plots generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()