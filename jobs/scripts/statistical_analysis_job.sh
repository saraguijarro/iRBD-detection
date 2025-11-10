#!/bin/bash
#BSUB -J statistical_analysis_irbd
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 6:00
#BSUB -o /work3/s184484/irbd-detection/jobs/logs/stats/stats_output_%J.out
#BSUB -e /work3/s184484/irbd-detection/jobs/logs/stats/stats_error_%J.err
#BSUB -N

# STATISTICAL ANALYSIS JOB SCRIPT FOR iRBD DETECTION PIPELINE
# Performs comprehensive statistical comparison of SSL-Wearables features between groups
# Analyzes 1024-dimensional features with multiple comparison correction

# =============================================================================
# JOB CONFIGURATION
# =============================================================================
echo "=========================================="
echo "iRBD Detection Pipeline - Statistical Analysis"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Cores: $LSB_DJOB_NUMPROC"
echo "Host: $LSB_HOSTS"
echo "Started: $(date)"
echo "=========================================="

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "Setting up environment..."

# Load conda module (DTU HPC specific)
module load miniconda3/4.12.0

# Activate statistical analysis environment
echo "Activating conda environment: env_insights"
source activate env_insights

# Verify environment and key packages
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check critical packages
python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "ERROR: numpy not found"
python -c "import scipy; print(f'scipy version: {scipy.__version__}')" || echo "ERROR: scipy not found"
python -c "import pandas; print(f'pandas version: {pandas.__version__}')" || echo "ERROR: pandas not found"
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')" || echo "ERROR: matplotlib not found"
python -c "import seaborn; print(f'seaborn version: {seaborn.__version__}')" || echo "ERROR: seaborn not found"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
echo "Setting up directories..."

# Main project directory
PROJECT_DIR="/work3/s184484/irbd-detection"
cd $PROJECT_DIR

# Verify input directories exist
if [ ! -d "data/features/controls" ]; then
    echo "ERROR: Controls features directory not found: data/features/controls"
    echo "Make sure feature extraction stage completed successfully"
    exit 1
fi

if [ ! -d "data/features/irbd" ]; then
    echo "ERROR: iRBD features directory not found: data/features/irbd"
    echo "Make sure feature extraction stage completed successfully"
    exit 1
fi

# Verify output directories exist (no creation - they should already exist)
if [ ! -d "results/statistical_analysis" ]; then
    echo "ERROR: Output directory not found: results/statistical_analysis"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "validation/data_quality_reports" ]; then
    echo "ERROR: Validation directory not found: validation/data_quality_reports"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "results/visualizations" ]; then
    echo "ERROR: Visualization directory not found: results/visualizations"
    echo "Make sure project structure is properly set up"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo "Working directory: $(pwd)"
echo "✓ All required directories verified"

# =============================================================================
# DATA VERIFICATION
# =============================================================================
echo "Verifying input data..."

# Count feature files
CONTROLS_COUNT=$(find data/features/controls -name "*_features.npy" | wc -l)
IRBD_COUNT=$(find data/features/irbd -name "*_features.npy" | wc -l)

echo "Feature files found:"
echo "  - Controls: $CONTROLS_COUNT .npy files"
echo "  - iRBD: $IRBD_COUNT .npy files"
echo "  - Total: $((CONTROLS_COUNT + IRBD_COUNT)) .npy files"

if [ $CONTROLS_COUNT -eq 0 ]; then
    echo "ERROR: No control feature files found"
    echo "Make sure feature extraction stage completed successfully"
    exit 1
fi

if [ $IRBD_COUNT -eq 0 ]; then
    echo "ERROR: No iRBD feature files found"
    echo "Make sure feature extraction stage completed successfully"
    exit 1
fi

# Check minimum sample size for statistical analysis
MIN_PARTICIPANTS=5
if [ $CONTROLS_COUNT -lt $MIN_PARTICIPANTS ]; then
    echo "ERROR: Insufficient control participants ($CONTROLS_COUNT < $MIN_PARTICIPANTS)"
    echo "Statistical analysis requires at least $MIN_PARTICIPANTS participants per group"
    exit 1
fi

if [ $IRBD_COUNT -lt $MIN_PARTICIPANTS ]; then
    echo "ERROR: Insufficient iRBD participants ($IRBD_COUNT < $MIN_PARTICIPANTS)"
    echo "Statistical analysis requires at least $MIN_PARTICIPANTS participants per group"
    exit 1
fi

echo "✓ Sufficient participants for statistical analysis"

# =============================================================================
# FEATURE DATA VERIFICATION
# =============================================================================
echo "Verifying feature data structure..."

# Test loading a sample file from each group
SAMPLE_CONTROL=$(find data/features/controls -name "*_features.npy" | head -1)
SAMPLE_IRBD=$(find data/features/irbd -name "*_features.npy" | head -1)

python -c "
import numpy as np
import sys

try:
    # Test control file
    control_data = np.load('$SAMPLE_CONTROL', allow_pickle=True).item()
    print(f'✓ Control sample loaded: $SAMPLE_CONTROL')
    print(f'  - Keys: {list(control_data.keys())}')
    
    if 'features' in control_data:
        control_features = control_data['features']
        print(f'  - Features shape: {control_features.shape}')
        if len(control_features.shape) >= 2 and control_features.shape[-1] == 1024:
            print(f'  - ✓ Correct feature dimension: 1024')
        else:
            print(f'  - ✗ Incorrect feature dimension: {control_features.shape[-1]}')
            sys.exit(1)
    
    # Test iRBD file
    irbd_data = np.load('$SAMPLE_IRBD', allow_pickle=True).item()
    print(f'✓ iRBD sample loaded: $SAMPLE_IRBD')
    print(f'  - Keys: {list(irbd_data.keys())}')
    
    if 'features' in irbd_data:
        irbd_features = irbd_data['features']
        print(f'  - Features shape: {irbd_features.shape}')
        if len(irbd_features.shape) >= 2 and irbd_features.shape[-1] == 1024:
            print(f'  - ✓ Correct feature dimension: 1024')
        else:
            print(f'  - ✗ Incorrect feature dimension: {irbd_features.shape[-1]}')
            sys.exit(1)
    
    print('✓ Feature data structure verified')
    
except Exception as e:
    print(f'✗ Feature data verification failed: {e}')
    sys.exit(1)
"

FEATURE_VERIFICATION_EXIT_CODE=$?
if [ $FEATURE_VERIFICATION_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Feature data verification failed"
    exit 1
fi

# =============================================================================
# STATISTICAL ANALYSIS EXECUTION
# =============================================================================
echo "Starting statistical analysis pipeline..."
echo "Timestamp: $(date)"
echo "Analyzing $((CONTROLS_COUNT + IRBD_COUNT)) participants with 1024-dimensional features"

# Record start time
START_TIME=$(date +%s)

# Run statistical analysis script
echo "Executing: python scripts/statistical_analysis.py"
python scripts/statistical_analysis.py

# Check exit status
STATISTICAL_ANALYSIS_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Statistical analysis completed with exit code: $STATISTICAL_ANALYSIS_EXIT_CODE"
echo "Total analysis time: $DURATION seconds ($(($DURATION / 60)) minutes)"

# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================
echo "Verifying statistical analysis outputs..."

# Check main results files
RESULTS_FILES=0
if [ -f "results/statistical_analysis/feature_analysis_summary.json" ]; then
    echo "✓ Analysis summary saved: feature_analysis_summary.json"
    RESULTS_FILES=$((RESULTS_FILES + 1))
else
    echo "✗ Missing analysis summary file: feature_analysis_summary.json"
fi

if [ -f "results/statistical_analysis/all_features_results.csv" ]; then
    echo "✓ All features results saved: all_features_results.csv"
    RESULTS_FILES=$((RESULTS_FILES + 1))
else
    echo "✗ Missing all features results file: all_features_results.csv"
fi

if [ -f "results/statistical_analysis/significant_features.csv" ]; then
    echo "✓ Significant features saved: significant_features.csv"
    RESULTS_FILES=$((RESULTS_FILES + 1))
else
    echo "✗ Missing significant features file: significant_features.csv"
fi

if [ -f "results/statistical_analysis/large_effect_features.csv" ]; then
    echo "✓ Large effect features saved: large_effect_features.csv"
    RESULTS_FILES=$((RESULTS_FILES + 1))
else
    echo "✗ Missing large effect features file: large_effect_features.csv"
fi

if [ -f "results/statistical_analysis/significant_and_large_effect_features.csv" ]; then
    echo "✓ Combined significant+large effect features saved"
    RESULTS_FILES=$((RESULTS_FILES + 1))
else
    echo "✗ Missing combined significant+large effect features file"
fi

echo "Results files: $RESULTS_FILES/5"

# =============================================================================
# QUALITY CHECKS
# =============================================================================
echo "Performing quality checks..."

# Check for log files
if [ -f "validation/data_quality_reports/statistical_analysis_"*".log" ]; then
    echo "✓ Statistical analysis log file created"
    LOG_FILE=$(ls -t validation/data_quality_reports/statistical_analysis_*.log | head -1)
    echo "Latest log: $LOG_FILE"
    
    # Show last few lines of log
    echo "Last 10 lines of statistical analysis log:"
    tail -10 "$LOG_FILE"
else
    echo "⚠ No statistical analysis log file found"
fi

# Check for visualization outputs
if [ -d "results/visualizations" ] && [ "$(ls -A results/visualizations)" ]; then
    PLOT_COUNT=$(find results/visualizations -name "*.png" | wc -l)
    echo "✓ $PLOT_COUNT visualization plots created"
else
    echo "⚠ No visualization plots found"
fi

# Verify statistical results (if summary exists)
if [ -f "results/statistical_analysis/feature_analysis_summary.json" ]; then
    echo "Verifying statistical results..."
    python -c "
import json
import sys

try:
    with open('results/statistical_analysis/feature_analysis_summary.json', 'r') as f:
        summary = json.load(f)
    
    print('✓ Statistical analysis summary:')
    print(f'  - Total features analyzed: {summary.get(\"total_features\", \"N/A\")}')
    print(f'  - Controls participants: {summary.get(\"n_controls\", \"N/A\")}')
    print(f'  - iRBD participants: {summary.get(\"n_irbd\", \"N/A\")}')
    print(f'  - Significant features (Bonferroni): {summary.get(\"n_significant_bonferroni\", \"N/A\")}')
    print(f'  - Large effect features (|d|≥0.5): {summary.get(\"n_large_effect\", \"N/A\")}')
    print(f'  - Combined significant+large: {summary.get(\"n_significant_and_large\", \"N/A\")}')
    
    # Check if we found meaningful differences
    n_significant = summary.get('n_significant_bonferroni', 0)
    n_large_effect = summary.get('n_large_effect', 0)
    
    if n_significant > 0:
        print(f'  - ✓ Found statistically significant group differences')
    else:
        print(f'  - ⚠ No statistically significant differences after correction')
    
    if n_large_effect > 0:
        print(f'  - ✓ Found clinically meaningful effect sizes')
    else:
        print(f'  - ⚠ No large effect sizes found')
    
    # Check analysis parameters
    if 'analysis_parameters' in summary:
        params = summary['analysis_parameters']
        print(f'  - Alpha level: {params.get(\"alpha\", \"N/A\")}')
        print(f'  - Effect size threshold: {params.get(\"effect_size_threshold\", \"N/A\")}')
        print(f'  - Multiple comparison method: {params.get(\"correction_method\", \"N/A\")}')
    
except Exception as e:
    print(f'⚠ Could not verify statistical results: {e}')
"
fi

# Verify CSV results files
if [ -f "results/statistical_analysis/all_features_results.csv" ]; then
    echo "Verifying CSV results structure..."
    python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('results/statistical_analysis/all_features_results.csv')
    print(f'✓ CSV results loaded successfully')
    print(f'  - Shape: {df.shape}')
    print(f'  - Columns: {list(df.columns)}')
    
    # Check expected columns
    expected_cols = ['feature_index', 'statistic', 'p_value', 'p_value_bonferroni', 
                    'effect_size', 'controls_mean', 'irbd_mean']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        print(f'  - ⚠ Missing columns: {missing_cols}')
    else:
        print(f'  - ✓ All expected columns present')
    
except Exception as e:
    print(f'⚠ Could not verify CSV results: {e}')
"
fi

# =============================================================================
# CLEANUP AND SUMMARY
# =============================================================================
echo "Cleaning up temporary files..."

# Remove any temporary files (if created)
find /tmp -name "*stats*" -user $(whoami) -delete 2>/dev/null || true

# =============================================================================
# JOB SUMMARY
# =============================================================================
echo "=========================================="
echo "STATISTICAL ANALYSIS JOB SUMMARY"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Exit Code: $STATISTICAL_ANALYSIS_EXIT_CODE"
echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
echo "Input Participants: $((CONTROLS_COUNT + IRBD_COUNT))"
echo "  - Controls: $CONTROLS_COUNT"
echo "  - iRBD: $IRBD_COUNT"
echo "Features Analyzed: 1024"
echo "Results Files: $RESULTS_FILES/5"
echo "Host: $LSB_HOSTS"
echo "Completed: $(date)"

# =============================================================================
# EXIT STATUS
# =============================================================================
if [ $STATISTICAL_ANALYSIS_EXIT_CODE -eq 0 ] && [ $RESULTS_FILES -ge 4 ]; then
    echo "✓ STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY"
    echo "Comprehensive group comparison analysis complete"
    exit 0
else
    echo "✗ STATISTICAL ANALYSIS FAILED"
    echo "Check error logs and analysis output"
    exit 1
fi

