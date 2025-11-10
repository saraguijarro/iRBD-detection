#!/bin/bash
#BSUB -J lstm_training_irbd
#BSUB -q gpua10
#BSUB -n 8
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -o /work3/s184484/irbd-detection/jobs/logs/lstm/lstm_output_%J.out
#BSUB -e /work3/s184484/irbd-detection/jobs/logs/lstm/lstm_error_%J.err
#BSUB -N

# LSTM TRAINING JOB SCRIPT FOR iRBD DETECTION PIPELINE
# Trains bidirectional LSTM with attention mechanism for iRBD classification
# Uses SSL-Wearables features from feature extraction stage

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
PROJECT_DIR="/work3/s184484/irbd-detection"
cd $PROJECT_DIR

# Verify input directories exist
if [ ! -d "data/features/combined" ]; then
    echo "ERROR: Combined features directory not found: data/features/combined"
    echo "Make sure feature extraction stage completed successfully"
    exit 1
fi

# Verify output directories exist (no creation - they should already exist)
if [ ! -d "results/lstm/models" ]; then
    echo "ERROR: Output directory not found: results/lstm/models"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "results/lstm/predictions" ]; then
    echo "ERROR: Output directory not found: results/lstm/predictions"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "results/lstm/evaluation" ]; then
    echo "ERROR: Output directory not found: results/lstm/evaluation"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "results/lstm/interpretability" ]; then
    echo "ERROR: Output directory not found: results/lstm/interpretability"
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

# Check required input files
REQUIRED_FILES=(
    "data/features/combined/X_features.npy"
    "data/features/combined/y_labels.npy"
    "data/features/combined/participant_ids.npy"
    "data/features/combined/dataset_info.json"
)

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo "ERROR: $MISSING_FILES required input files missing"
    echo "Make sure feature extraction stage completed successfully"
    exit 1
fi

# Verify data structure
echo "Verifying data structure..."
python -c "
import numpy as np
import json
import sys

try:
    # Load data
    X = np.load('data/features/combined/X_features.npy')
    y = np.load('data/features/combined/y_labels.npy')
    ids = np.load('data/features/combined/participant_ids.npy')
    
    with open('data/features/combined/dataset_info.json', 'r') as f:
        info = json.load(f)
    
    print(f'✓ Data loaded successfully')
    print(f'  - Features shape: {X.shape}')
    print(f'  - Labels shape: {y.shape}')
    print(f'  - Participant IDs shape: {ids.shape}')
    print(f'  - Total windows: {X.shape[0]:,}')
    print(f'  - Feature dimension: {X.shape[1]}')
    print(f'  - Controls: {np.sum(y == 0):,} windows ({np.sum(y == 0)/len(y)*100:.1f}%)')
    print(f'  - iRBD: {np.sum(y == 1):,} windows ({np.sum(y == 1)/len(y)*100:.1f}%)')
    print(f'  - Unique participants: {len(np.unique(ids))}')
    
    # Verify data integrity
    if X.shape[0] != y.shape[0] or X.shape[0] != ids.shape[0]:
        print('✗ Data shape mismatch')
        sys.exit(1)
        
    if X.shape[1] != 1024:
        print(f'✗ Incorrect feature dimension: {X.shape[1]} (expected 1024)')
        sys.exit(1)
        
    if not set(np.unique(y)).issubset({0, 1}):
        print(f'✗ Invalid labels: {np.unique(y)} (expected 0 and 1)')
        sys.exit(1)
        
    print('✓ Data integrity verified')
    
except Exception as e:
    print(f'✗ Data verification failed: {e}')
    sys.exit(1)
"

DATA_VERIFICATION_EXIT_CODE=$?
if [ $DATA_VERIFICATION_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Data verification failed"
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
echo "Executing: python scripts/lstm.py"
python scripts/lstm.py

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

# Check model files
MODEL_FILES=0
if [ -f "results/lstm/models/best_lstm_model.pth" ]; then
    echo "✓ Best model saved: best_lstm_model.pth"
    MODEL_FILES=$((MODEL_FILES + 1))
else
    echo "✗ Missing best model file: best_lstm_model.pth"
fi

if [ -f "results/lstm/models/final_lstm_model.pth" ]; then
    echo "✓ Final model saved: final_lstm_model.pth"
    MODEL_FILES=$((MODEL_FILES + 1))
else
    echo "✗ Missing final model file: final_lstm_model.pth"
fi

if [ -f "results/lstm/models/training_history.json" ]; then
    echo "✓ Training history saved: training_history.json"
    MODEL_FILES=$((MODEL_FILES + 1))
else
    echo "✗ Missing training history file: training_history.json"
fi

# Check prediction files
PREDICTION_FILES=0
if [ -f "results/lstm/predictions/test_predictions.npy" ]; then
    echo "✓ Test predictions saved: test_predictions.npy"
    PREDICTION_FILES=$((PREDICTION_FILES + 1))
else
    echo "✗ Missing test predictions file: test_predictions.npy"
fi

if [ -f "results/lstm/predictions/participant_predictions.json" ]; then
    echo "✓ Participant predictions saved: participant_predictions.json"
    PREDICTION_FILES=$((PREDICTION_FILES + 1))
else
    echo "✗ Missing participant predictions file: participant_predictions.json"
fi

# Check evaluation files
EVALUATION_FILES=0
if [ -f "results/lstm/evaluation/evaluation_summary.json" ]; then
    echo "✓ Evaluation summary saved: evaluation_summary.json"
    EVALUATION_FILES=$((EVALUATION_FILES + 1))
else
    echo "✗ Missing evaluation summary file: evaluation_summary.json"
fi

if [ -f "results/lstm/evaluation/confusion_matrices.json" ]; then
    echo "✓ Confusion matrices saved: confusion_matrices.json"
    EVALUATION_FILES=$((EVALUATION_FILES + 1))
else
    echo "✗ Missing confusion matrices file: confusion_matrices.json"
fi

if [ -f "results/lstm/evaluation/cross_validation_results.json" ]; then
    echo "✓ Cross-validation results saved: cross_validation_results.json"
    EVALUATION_FILES=$((EVALUATION_FILES + 1))
else
    echo "✗ Missing cross-validation results file: cross_validation_results.json"
fi

# Check interpretability files
INTERPRETABILITY_FILES=0
if [ -f "results/lstm/interpretability/attention_weights.npy" ]; then
    echo "✓ Attention weights saved: attention_weights.npy"
    INTERPRETABILITY_FILES=$((INTERPRETABILITY_FILES + 1))
else
    echo "✗ Missing attention weights file: attention_weights.npy"
fi

if [ -f "results/lstm/interpretability/feature_importance.json" ]; then
    echo "✓ Feature importance saved: feature_importance.json"
    INTERPRETABILITY_FILES=$((INTERPRETABILITY_FILES + 1))
else
    echo "✗ Missing feature importance file: feature_importance.json"
fi

TOTAL_OUTPUT_FILES=$((MODEL_FILES + PREDICTION_FILES + EVALUATION_FILES + INTERPRETABILITY_FILES))
echo "Total output files: $TOTAL_OUTPUT_FILES/9"

# =============================================================================
# QUALITY CHECKS
# =============================================================================
echo "Performing quality checks..."

# Check for log files
if [ -f "validation/data_quality_reports/lstm_training_"*".log" ]; then
    echo "✓ LSTM training log file created"
    LOG_FILE=$(ls -t validation/data_quality_reports/lstm_training_*.log | head -1)
    echo "Latest log: $LOG_FILE"
    
    # Show last few lines of log
    echo "Last 10 lines of LSTM training log:"
    tail -10 "$LOG_FILE"
else
    echo "⚠ No LSTM training log file found"
fi

# Check for visualization outputs
if [ -d "results/visualizations" ] && [ "$(ls -A results/visualizations)" ]; then
    PLOT_COUNT=$(find results/visualizations -name "*.png" | wc -l)
    echo "✓ $PLOT_COUNT visualization plots created"
else
    echo "⚠ No visualization plots found"
fi

# Verify model performance (if evaluation summary exists)
if [ -f "results/lstm/evaluation/evaluation_summary.json" ]; then
    echo "Verifying model performance..."
    python -c "
import json
import sys

try:
    with open('results/lstm/evaluation/evaluation_summary.json', 'r') as f:
        eval_summary = json.load(f)
    
    print('✓ Model performance summary:')
    
    # Check if we have test performance
    if 'test_performance' in eval_summary:
        test_perf = eval_summary['test_performance']
        print(f'  - Test Accuracy: {test_perf.get(\"accuracy\", \"N/A\"):.3f}')
        print(f'  - Test Sensitivity: {test_perf.get(\"sensitivity\", \"N/A\"):.3f}')
        print(f'  - Test Specificity: {test_perf.get(\"specificity\", \"N/A\"):.3f}')
        print(f'  - Test AUC: {test_perf.get(\"auc\", \"N/A\"):.3f}')
        
        # Check if sensitivity meets clinical target (≥0.85)
        sensitivity = test_perf.get('sensitivity', 0)
        if sensitivity >= 0.85:
            print(f'  - ✓ Clinical sensitivity target met (≥0.85)')
        else:
            print(f'  - ⚠ Clinical sensitivity target not met (<0.85)')
    
    # Check cross-validation performance
    if 'cv_performance' in eval_summary:
        cv_perf = eval_summary['cv_performance']
        print(f'  - CV Mean Accuracy: {cv_perf.get(\"mean_accuracy\", \"N/A\"):.3f} ± {cv_perf.get(\"std_accuracy\", \"N/A\"):.3f}')
        print(f'  - CV Mean AUC: {cv_perf.get(\"mean_auc\", \"N/A\"):.3f} ± {cv_perf.get(\"std_auc\", \"N/A\"):.3f}')
    
except Exception as e:
    print(f'⚠ Could not verify model performance: {e}')
"
fi

# =============================================================================
# CLEANUP AND SUMMARY
# =============================================================================
echo "Cleaning up temporary files..."

# Clear GPU memory
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU memory cleared')
"

# Remove any temporary files (if created)
find /tmp -name "*lstm*" -user $(whoami) -delete 2>/dev/null || true

# =============================================================================
# JOB SUMMARY
# =============================================================================
echo "=========================================="
echo "LSTM TRAINING JOB SUMMARY"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Exit Code: $LSTM_TRAINING_EXIT_CODE"
echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
echo "Output Files: $TOTAL_OUTPUT_FILES/9"
echo "  - Models: $MODEL_FILES/3"
echo "  - Predictions: $PREDICTION_FILES/2"
echo "  - Evaluation: $EVALUATION_FILES/3"
echo "  - Interpretability: $INTERPRETABILITY_FILES/2"
echo "Host: $LSB_HOSTS"
echo "GPU: $LSB_GPU_REQ"
echo "Completed: $(date)"

# =============================================================================
# EXIT STATUS
# =============================================================================
if [ $LSTM_TRAINING_EXIT_CODE -eq 0 ] && [ $TOTAL_OUTPUT_FILES -ge 7 ]; then
    echo "✓ LSTM TRAINING COMPLETED SUCCESSFULLY"
    echo "Model trained and evaluated - ready for clinical validation"
    exit 0
else
    echo "✗ LSTM TRAINING FAILED"
    echo "Check error logs and training output"
    exit 1
fi

