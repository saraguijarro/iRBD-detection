#!/bin/bash
#BSUB -J ml_v1
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /work3/s184484/iRBD-detection/jobs/logs/ml/ml_v1_%J.out
#BSUB -e /work3/s184484/iRBD-detection/jobs/logs/ml/ml_v1_%J.err
#BSUB -N

# ML BASELINES JOB SCRIPT FOR iRBD DETECTION PIPELINE - VERSION V1
# Trains classical ML models (Random Forest, Logistic Regression, SVM, XGBoost)
# Uses participant-level aggregated SSL-Wearables features
# V1: Fixed nights + temperature filter

# =============================================================================
# JOB CONFIGURATION
# =============================================================================
echo "=========================================="
echo "iRBD Detection Pipeline - ML Baselines"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Cores: $LSB_DJOB_NUMPROC"
echo "Host: $LSB_HOSTS"
echo "Started: $(date)"
echo ""
echo "Preprocessing version: v1 (Fixed nights + temp filter)"
echo "=========================================="

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "Setting up environment..."

# Load conda module (DTU HPC specific)
module load miniconda3/4.12.0

# Activate ML environment
echo "Activating conda environment: env_insights"
source activate env_insights

# Verify environment and key packages
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check critical packages
python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "ERROR: numpy not found"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')" || echo "ERROR: scikit-learn not found"
python -c "import xgboost; print(f'xgboost version: {xgboost.__version__}')" || echo "ERROR: xgboost not found"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
echo "Setting up directories..."

# Main project directory
PROJECT_DIR="/work3/s184484/iRBD-detection"
cd $PROJECT_DIR

# Feature directory for V1
FEATURES_DIR="data/features_v1"

echo "Project directory: $PROJECT_DIR"
echo "Features directory: $FEATURES_DIR"
echo "Working directory: $(pwd)"

# Verify input directories exist
if [ ! -d "$FEATURES_DIR/controls" ]; then
    echo "ERROR: Controls features directory not found: $FEATURES_DIR/controls"
    echo "Make sure feature extraction stage completed successfully for v1"
    exit 1
fi

if [ ! -d "$FEATURES_DIR/irbd" ]; then
    echo "ERROR: iRBD features directory not found: $FEATURES_DIR/irbd"
    echo "Make sure feature extraction stage completed successfully for v1"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p results/ml_baselines_v1

echo "All required directories verified/created"

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
# ML BASELINES EXECUTION
# =============================================================================
echo "Starting ML baselines pipeline..."
echo "Timestamp: $(date)"

# Record start time
START_TIME=$(date +%s)

# Run ML baselines script
echo "Executing: python scripts/ml_baselines.py --version v1"
python scripts/ml_baselines.py --version v1

# Check exit status
ML_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "ML baselines completed with exit code: $ML_EXIT_CODE"
echo "Total training time: $DURATION seconds ($(($DURATION / 60)) minutes)"

# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================
echo "Verifying ML baselines outputs..."

# Check for output files in results/ml_baselines_{version}
if [ -d "results/ml_baselines_v1" ]; then
    ML_COUNT=$(find results/ml_baselines_v1 -type f 2>/dev/null | wc -l)
    echo "ML results files: $ML_COUNT"
else
    ML_COUNT=0
    echo "WARNING: Output directory not found"
fi

if [ $ML_COUNT -eq 0 ]; then
    echo "WARNING: No output files generated"
fi


# =============================================================================
# JOB SUMMARY
# =============================================================================
echo "=========================================="
echo "JOB SUMMARY"
echo "=========================================="
echo "Version: v1 (Fixed nights + temp filter)"
echo "Input features: $FEATURES_DIR"
echo "Participants: $TOTAL_COUNT ($CONTROLS_COUNT controls + $IRBD_COUNT iRBD)"
echo "Training time: $(($DURATION / 60)) minutes"
echo "Exit code: $ML_EXIT_CODE"
echo "Output files: $ML_COUNT"
echo "Completed: $(date)"
echo "=========================================="

# Exit with ML baselines exit code
exit $ML_EXIT_CODE
