#!/bin/bash
#BSUB -J feature_extraction_irbd
#BSUB -q gpua10
#BSUB -n 8
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /work3/s184484/irbd-detection/jobs/logs/feature_extraction/feature_extraction_output_%J.out
#BSUB -e /work3/s184484/irbd-detection/jobs/logs/feature_extraction/feature_extraction_error_%J.err
#BSUB -N

# FEATURE EXTRACTION JOB SCRIPT FOR iRBD DETECTION PIPELINE
# Extracts 1024-dimensional SSL-Wearables features from preprocessed accelerometer data
# Uses harnet10 model from OxWearables/ssl-wearables

# =============================================================================
# JOB CONFIGURATION
# =============================================================================
echo "=========================================="
echo "iRBD Detection Pipeline - Feature Extraction"
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

# Activate feature extraction environment
echo "Activating conda environment: env_insights"
source activate env_insights

# Verify environment and key packages
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check critical packages
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "ERROR: PyTorch not found"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "WARNING: CUDA check failed"
python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "ERROR: numpy not found"
python -c "import h5py; print(f'h5py version: {h5py.__version__}')" || echo "ERROR: h5py not found"

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
if [ ! -d "data/preprocessed/controls" ]; then
    echo "ERROR: Preprocessed controls directory not found: data/preprocessed/controls"
    exit 1
fi

if [ ! -d "data/preprocessed/irbd" ]; then
    echo "ERROR: Preprocessed iRBD directory not found: data/preprocessed/irbd"
    exit 1
fi

# Verify output directories exist (no creation - they should already exist)
if [ ! -d "data/features/controls" ]; then
    echo "ERROR: Output directory not found: data/features/controls"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "data/features/irbd" ]; then
    echo "ERROR: Output directory not found: data/features/irbd"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "data/features/combined" ]; then
    echo "ERROR: Output directory not found: data/features/combined"
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

# Count input files
CONTROLS_COUNT=$(find data/preprocessed/controls -name "*.h5" | wc -l)
IRBD_COUNT=$(find data/preprocessed/irbd -name "*.h5" | wc -l)

echo "Input files found:"
echo "  - Controls: $CONTROLS_COUNT .h5 files"
echo "  - iRBD: $IRBD_COUNT .h5 files"
echo "  - Total: $((CONTROLS_COUNT + IRBD_COUNT)) .h5 files"

if [ $CONTROLS_COUNT -eq 0 ]; then
    echo "ERROR: No preprocessed control .h5 files found"
    echo "Make sure preprocessing stage completed successfully"
    exit 1
fi

if [ $IRBD_COUNT -eq 0 ]; then
    echo "ERROR: No preprocessed iRBD .h5 files found"
    echo "Make sure preprocessing stage completed successfully"
    exit 1
fi

# =============================================================================
# SSL-WEARABLES MODEL SETUP
# =============================================================================
echo "Setting up SSL-Wearables model..."

# Test model loading
python -c "
import torch
import torch.hub
print('Testing SSL-Wearables model loading...')
try:
    # Load model (this will download if not cached)
    model = torch.hub.load('OxWearables/ssl-wearables', 'harnet10', 
                          class_num=5, pretrained=True, trust_repo=True)
    print('✓ SSL-Wearables harnet10 model loaded successfully')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test GPU transfer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'✓ Model transferred to: {device}')
    
except Exception as e:
    print(f'✗ Model loading failed: {e}')
    exit(1)
"

MODEL_SETUP_EXIT_CODE=$?
if [ $MODEL_SETUP_EXIT_CODE -ne 0 ]; then
    echo "ERROR: SSL-Wearables model setup failed"
    exit 1
fi

# =============================================================================
# FEATURE EXTRACTION EXECUTION
# =============================================================================
echo "Starting feature extraction pipeline..."
echo "Timestamp: $(date)"

# Record start time
START_TIME=$(date +%s)

# Run feature extraction script
echo "Executing: python scripts/feature_extraction.py"
python scripts/feature_extraction.py

# Check exit status
FEATURE_EXTRACTION_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Feature extraction completed with exit code: $FEATURE_EXTRACTION_EXIT_CODE"
echo "Total processing time: $DURATION seconds ($(($DURATION / 60)) minutes)"

# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================
echo "Verifying feature extraction outputs..."

# Count individual feature files
CONTROLS_FEATURES=$(find data/features/controls -name "*_features.npy" | wc -l)
IRBD_FEATURES=$(find data/features/irbd -name "*_features.npy" | wc -l)

echo "Individual feature files created:"
echo "  - Controls: $CONTROLS_FEATURES .npy files"
echo "  - iRBD: $IRBD_FEATURES .npy files"
echo "  - Total: $((CONTROLS_FEATURES + IRBD_FEATURES)) .npy files"

# Check combined dataset files
COMBINED_FILES=0
if [ -f "data/features/combined/X_features.npy" ]; then
    echo "✓ Combined features file created: X_features.npy"
    COMBINED_FILES=$((COMBINED_FILES + 1))
else
    echo "✗ Missing combined features file: X_features.npy"
fi

if [ -f "data/features/combined/y_labels.npy" ]; then
    echo "✓ Combined labels file created: y_labels.npy"
    COMBINED_FILES=$((COMBINED_FILES + 1))
else
    echo "✗ Missing combined labels file: y_labels.npy"
fi

if [ -f "data/features/combined/participant_ids.npy" ]; then
    echo "✓ Participant mapping file created: participant_ids.npy"
    COMBINED_FILES=$((COMBINED_FILES + 1))
else
    echo "✗ Missing participant mapping file: participant_ids.npy"
fi

if [ -f "data/features/combined/dataset_info.json" ]; then
    echo "✓ Dataset metadata file created: dataset_info.json"
    COMBINED_FILES=$((COMBINED_FILES + 1))
else
    echo "✗ Missing dataset metadata file: dataset_info.json"
fi

# Calculate success rate
TOTAL_INPUT=$((CONTROLS_COUNT + IRBD_COUNT))
TOTAL_OUTPUT=$((CONTROLS_FEATURES + IRBD_FEATURES))
SUCCESS_RATE=$(echo "scale=2; $TOTAL_OUTPUT * 100 / $TOTAL_INPUT" | bc)

echo "Processing success rate: $SUCCESS_RATE% ($TOTAL_OUTPUT/$TOTAL_INPUT)"

# =============================================================================
# QUALITY CHECKS
# =============================================================================
echo "Performing quality checks..."

# Check for log files
if [ -f "validation/data_quality_reports/feature_extraction_"*".log" ]; then
    echo "✓ Feature extraction log file created"
    LOG_FILE=$(ls -t validation/data_quality_reports/feature_extraction_*.log | head -1)
    echo "Latest log: $LOG_FILE"
    
    # Show last few lines of log
    echo "Last 10 lines of feature extraction log:"
    tail -10 "$LOG_FILE"
else
    echo "⚠ No feature extraction log file found"
fi

# Check for visualization outputs
if [ -d "results/visualizations" ] && [ "$(ls -A results/visualizations)" ]; then
    PLOT_COUNT=$(find results/visualizations -name "*.png" | wc -l)
    echo "✓ $PLOT_COUNT visualization plots created"
else
    echo "⚠ No visualization plots found"
fi

# Verify feature file structure (sample check)
if [ $TOTAL_OUTPUT -gt 0 ]; then
    echo "Verifying feature file structure..."
    SAMPLE_NPY=$(find data/features -name "*_features.npy" | head -1)
    if [ -n "$SAMPLE_NPY" ]; then
        echo "Sample file: $SAMPLE_NPY"
        python -c "
import numpy as np
import sys
try:
    data = np.load('$SAMPLE_NPY', allow_pickle=True).item()
    print(f'✓ Feature file structure valid')
    print(f'  - Keys: {list(data.keys())}')
    if 'features' in data:
        features_shape = data['features'].shape
        print(f'  - Features shape: {features_shape}')
        if len(features_shape) >= 2 and features_shape[-1] == 1024:
            print(f'  - ✓ Correct feature dimension: 1024')
        else:
            print(f'  - ✗ Incorrect feature dimension: {features_shape[-1]}')
            sys.exit(1)
    if 'participant_id' in data:
        print(f'  - Participant: {data[\"participant_id\"]}')
except Exception as e:
    print(f'✗ Feature file structure error: {e}')
    sys.exit(1)
"
    fi
fi

# Verify combined dataset structure
if [ $COMBINED_FILES -eq 4 ]; then
    echo "Verifying combined dataset structure..."
    python -c "
import numpy as np
import json
import sys
try:
    # Load combined files
    X = np.load('data/features/combined/X_features.npy')
    y = np.load('data/features/combined/y_labels.npy')
    ids = np.load('data/features/combined/participant_ids.npy')
    
    with open('data/features/combined/dataset_info.json', 'r') as f:
        info = json.load(f)
    
    print(f'✓ Combined dataset structure valid')
    print(f'  - Features shape: {X.shape}')
    print(f'  - Labels shape: {y.shape}')
    print(f'  - Participant IDs shape: {ids.shape}')
    print(f'  - Total windows: {X.shape[0]:,}')
    print(f'  - Feature dimension: {X.shape[1]}')
    print(f'  - Controls: {np.sum(y == 0):,} windows')
    print(f'  - iRBD: {np.sum(y == 1):,} windows')
    
    # Verify dimensions match
    if X.shape[0] == y.shape[0] == ids.shape[0]:
        print(f'  - ✓ All arrays have matching lengths')
    else:
        print(f'  - ✗ Array length mismatch')
        sys.exit(1)
        
    if X.shape[1] == 1024:
        print(f'  - ✓ Correct feature dimension: 1024')
    else:
        print(f'  - ✗ Incorrect feature dimension: {X.shape[1]}')
        sys.exit(1)
        
except Exception as e:
    print(f'✗ Combined dataset structure error: {e}')
    sys.exit(1)
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
find /tmp -name "*ssl_wearables*" -user $(whoami) -delete 2>/dev/null || true

# =============================================================================
# JOB SUMMARY
# =============================================================================
echo "=========================================="
echo "FEATURE EXTRACTION JOB SUMMARY"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Exit Code: $FEATURE_EXTRACTION_EXIT_CODE"
echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
echo "Input Files: $TOTAL_INPUT (.h5 files)"
echo "Output Files: $TOTAL_OUTPUT (.npy files)"
echo "Combined Files: $COMBINED_FILES/4 files"
echo "Success Rate: $SUCCESS_RATE%"
echo "Host: $LSB_HOSTS"
echo "GPU: $LSB_GPU_REQ"
echo "Completed: $(date)"

# =============================================================================
# EXIT STATUS
# =============================================================================
if [ $FEATURE_EXTRACTION_EXIT_CODE -eq 0 ] && [ $TOTAL_OUTPUT -gt 0 ] && [ $COMBINED_FILES -eq 4 ]; then
    echo "✓ FEATURE EXTRACTION COMPLETED SUCCESSFULLY"
    echo "Ready for LSTM training and statistical analysis stages"
    exit 0
else
    echo "✗ FEATURE EXTRACTION FAILED"
    echo "Check error logs and feature extraction output"
    exit 1
fi

