#!/bin/bash
#BSUB -J lstm_v1
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /work3/s184484/iRBD-detection/jobs/logs/lstm/lstm_v1_%J.out
#BSUB -e /work3/s184484/iRBD-detection/jobs/logs/lstm/lstm_v1_%J.err
#BSUB -N

# LSTM TRAINING JOB SCRIPT FOR iRBD DETECTION PIPELINE - VERSION V1
# Trains bidirectional LSTM with attention mechanism for iRBD classification
# Uses SSL-Wearables features from feature extraction stage
# V1: Fixed nights + temperature filter

# =============================================================================
# JOB CONFIGURATION
# =============================================================================
echo "=========================================="
echo "iRBD Detection Pipeline - LSTM Training"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Cores: $LSB_DJOB_NUMPROC"
echo "GPU: $LSB_GPU_REQ"
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

# Activate LSTM training environment
echo "Activating conda environment: env_insights"
source activate env_insights

# Verify environment and key packages
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check critical packages
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "ERROR: PyTorch not found"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "WARNING: CUDA check failed"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')" || echo "ERROR: scikit-learn not found"
python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "ERROR: numpy not found"

# =============================================================================
# GPU VERIFICATION
# =============================================================================
echo "Verifying GPU setup..."

# Check GPU availability
nvidia-smi || echo "WARNING: nvidia-smi not available"

# Check PyTorch GPU access
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available - will use CPU (very slow)')
"

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
mkdir -p results/lstm_v1/models
mkdir -p results/lstm_v1/predictions
mkdir -p results/lstm_v1/evaluation
mkdir -p results/lstm_v1/interpretability
mkdir -p results/visualizations/v1

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
# LSTM TRAINING EXECUTION
# =============================================================================
echo "Starting LSTM training pipeline..."
echo "Timestamp: $(date)"

# Record start time
START_TIME=$(date +%s)

# Run LSTM training script
echo "Executing: python scripts/lstm.py --version v1"
python scripts/lstm.py --version v1

# Check exit status
LSTM_TRAINING_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "LSTM training completed with exit code: $LSTM_TRAINING_EXIT_CODE"
echo "Total training time: $DURATION seconds ($(($DURATION / 60)) minutes)"

# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================
echo "Verifying LSTM training outputs..."

# Check for any output files in results/lstm_v1
TOTAL_OUTPUT_FILES=0

# Check models
if [ -d "results/lstm_v1/models" ]; then
    MODEL_COUNT=$(find results/lstm_v1/models -type f 2>/dev/null | wc -l)
    echo "Model files: $MODEL_COUNT"
    TOTAL_OUTPUT_FILES=$((TOTAL_OUTPUT_FILES + MODEL_COUNT))
fi

# Check predictions
if [ -d "results/lstm_v1/predictions" ]; then
    PRED_COUNT=$(find results/lstm_v1/predictions -type f 2>/dev/null | wc -l)
    echo "Prediction files: $PRED_COUNT"
    TOTAL_OUTPUT_FILES=$((TOTAL_OUTPUT_FILES + PRED_COUNT))
fi

# Check evaluation
if [ -d "results/lstm_v1/evaluation" ]; then
    EVAL_COUNT=$(find results/lstm_v1/evaluation -type f 2>/dev/null | wc -l)
    echo "Evaluation files: $EVAL_COUNT"
    TOTAL_OUTPUT_FILES=$((TOTAL_OUTPUT_FILES + EVAL_COUNT))
fi

# Check interpretability
if [ -d "results/lstm_v1/interpretability" ]; then
    INTERP_COUNT=$(find results/lstm_v1/interpretability -type f 2>/dev/null | wc -l)
    echo "Interpretability files: $INTERP_COUNT"
    TOTAL_OUTPUT_FILES=$((TOTAL_OUTPUT_FILES + INTERP_COUNT))
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
echo "Version: v1 (Fixed nights + temp filter)"
echo "Input features: $FEATURES_DIR"
echo "Participants: $TOTAL_COUNT ($CONTROLS_COUNT controls + $IRBD_COUNT iRBD)"
echo "Training time: $(($DURATION / 60)) minutes"
echo "Exit code: $LSTM_TRAINING_EXIT_CODE"
echo "Output files: $TOTAL_OUTPUT_FILES"
echo "Completed: $(date)"
echo "=========================================="

# Exit with LSTM training exit code
exit $LSTM_TRAINING_EXIT_CODE