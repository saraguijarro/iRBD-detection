#!/bin/bash
#BSUB -J stats_vvt
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 6:00
#BSUB -o /work3/s184484/iRBD-detection/jobs/logs/stats/stats_vvt_%J.out
#BSUB -e /work3/s184484/iRBD-detection/jobs/logs/stats/stats_vvt_%J.err
#BSUB -N

# STATISTICAL ANALYSIS JOB SCRIPT FOR iRBD DETECTION PIPELINE - VERSION vvt
# Performs comprehensive statistical analysis of SSL-Wearables features
# Compares iRBD patients vs healthy controls
# vvt: Fixed nights + temperature filter

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
echo ""
echo "Preprocessing version: vvt (Fixed nights + temp filter)"
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
python -c "import pandas; print(f'pandas version: {pandas.__version__}')" || echo "ERROR: pandas not found"
python -c "import scipy; print(f'scipy version: {scipy.__version__}')" || echo "ERROR: scipy not found"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')" || echo "ERROR: scikit-learn not found"
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')" || echo "ERROR: matplotlib not found"
python -c "import seaborn; print(f'seaborn version: {seaborn.__version__}')" || echo "ERROR: seaborn not found"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
echo "Setting up directories..."

# Main project directory
PROJECT_DIR="/work3/s184484/iRBD-detection"
cd $PROJECT_DIR

# Feature directory for vvt
FEATURES_DIR="data/features_vvt"

echo "Project directory: $PROJECT_DIR"
echo "Features directory: $FEATURES_DIR"
echo "Working directory: $(pwd)"

# Verify input directories exist
if [ ! -d "$FEATURES_DIR/controls" ]; then
    echo "ERROR: Controls features directory not found: $FEATURES_DIR/controls"
    echo "Make sure feature extraction stage completed successfully for vvt"
    exit 1
fi

if [ ! -d "$FEATURES_DIR/irbd" ]; then
    echo "ERROR: iRBD features directory not found: $FEATURES_DIR/irbd"
    echo "Make sure feature extraction stage completed successfully for vvt"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p results/statistical_analysis_vvt
mkdir -p results/visualizations/vvt
mkdir -p validation/data_quality_reports

echo "✓ All required directories verified/created"

# =============================================================================
# DATA VERIFICATION
# =============================================================================
echo "Verifying input data..."

# Count feature files
CONTROLS_COUNT=$(find $FEATURES_DIR/controls -name "*.npz" 2>/dev/null | wc -l)
IRBD_COUNT=$(find $FEATURES_DIR/irbd -name "*.npz" 2>/dev/null | wc -l)
TOTAL_COUNT=$((CONTROLS_COUNT + IRBD_COUNT))

echo "Feature files found:"
echo "  Controls: $CONTROLS_COUNT"
echo "  iRBD: $IRBD_COUNT"
echo "  Total: $TOTAL_COUNT"

if [ $TOTAL_COUNT -eq 0 ]; then
    echo "ERROR: No feature files found in $FEATURES_DIR"
    echo "Make sure feature extraction stage completed successfully"
    exit 1
fi

if [ $CONTROLS_COUNT -eq 0 ]; then
    echo "ERROR: No control feature files found"
    exit 1
fi

if [ $IRBD_COUNT -eq 0 ]; then
    echo "ERROR: No iRBD feature files found"
    exit 1
fi

# =============================================================================
# STATISTICAL ANALYSIS EXECUTION
# =============================================================================
echo "Starting statistical analysis pipeline..."
echo "Timestamp: $(date)"

# Record start time
START_TIME=$(date +%s)

# Run statistical analysis script
echo "Executing: python scripts/statistical_analysis.py --version vvt"
python scripts/statistical_analysis.py --version vvt

# Check exit status
STATS_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Statistical analysis completed with exit code: $STATS_EXIT_CODE"
echo "Total analysis time: $DURATION seconds ($(($DURATION / 60)) minutes)"

# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================
echo "Verifying statistical analysis outputs..."

# Check for output files in results/statistical_analysis_vvt
TOTAL_OUTPUT_FILES=0

if [ -d "results/statistical_analysis_vvt" ]; then
    STATS_COUNT=$(find results/statistical_analysis_vvt -type f 2>/dev/null | wc -l)
    echo "Statistical results files: $STATS_COUNT"
    TOTAL_OUTPUT_FILES=$((TOTAL_OUTPUT_FILES + STATS_COUNT))
fi

if [ -d "results/visualizations/vvt" ]; then
    VIZ_COUNT=$(find results/visualizations/vvt -type f 2>/dev/null | wc -l)
    echo "Visualization files: $VIZ_COUNT"
    TOTAL_OUTPUT_FILES=$((TOTAL_OUTPUT_FILES + VIZ_COUNT))
fi

echo "Total output files: $TOTAL_OUTPUT_FILES"

if [ $TOTAL_OUTPUT_FILES -eq 0 ]; then
    echo "WARNING: No output files generated"
fi

# =============================================================================
# JOB SUMMARY
# =============================================================================
echo "=========================================="
echo "JOB SUMMARY"
echo "=========================================="
echo "Version: vvt (Fixed nights + temp filter)"
echo "Input features: $FEATURES_DIR"
echo "Participants: $TOTAL_COUNT ($CONTROLS_COUNT controls + $IRBD_COUNT iRBD)"
echo "Analysis time: $(($DURATION / 60)) minutes"
echo "Exit code: $STATS_EXIT_CODE"
echo "Output files: $TOTAL_OUTPUT_FILES"
echo "Completed: $(date)"
echo "=========================================="

# Exit with statistical analysis exit code
exit $STATS_EXIT_CODE
