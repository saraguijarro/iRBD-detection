#!/bin/bash
#BSUB -J preprocessing_irbd
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /work3/s184484/irbd-detection/jobs/logs/preprocessing/preprocessing_output_%J.out
#BSUB -e /work3/s184484/irbd-detection/jobs/logs/preprocessing/preprocessing_error_%J.err
#BSUB -N

# PREPROCESSING JOB SCRIPT FOR iRBD DETECTION PIPELINE
# Processes raw .cwa accelerometer files into clean night-segmented data
# Uses actipy for reading, filtering, and quality control

# =============================================================================
# JOB CONFIGURATION
# =============================================================================
echo "=========================================="
echo "iRBD Detection Pipeline - Preprocessing"
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

# Activate preprocessing environment
echo "Activating conda environment: env_preprocessing"
source activate env_preprocessing

# Verify environment and key packages
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check critical packages
python -c "import actipy; print(f'actipy version: {actipy.__version__}')" || echo "WARNING: actipy not found"
python -c "import h5py; print(f'h5py version: {h5py.__version__}')" || echo "WARNING: h5py not found"
python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "WARNING: numpy not found"
python -c "import pandas; print(f'pandas version: {pandas.__version__}')" || echo "WARNING: pandas not found"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
echo "Setting up directories..."

# Main project directory
PROJECT_DIR="/work3/s184484/irbd-detection"
cd $PROJECT_DIR

# Verify critical directories exist
if [ ! -d "data/raw/controls" ]; then
    echo "ERROR: Raw controls directory not found: data/raw/controls"
    exit 1
fi

if [ ! -d "data/raw/irbd" ]; then
    echo "ERROR: Raw iRBD directory not found: data/raw/irbd"
    exit 1
fi

# Verify output directories exist (no creation - they should already exist)
if [ ! -d "data/preprocessed/controls" ]; then
    echo "ERROR: Output directory not found: data/preprocessed/controls"
    echo "Make sure project structure is properly set up"
    exit 1
fi

if [ ! -d "data/preprocessed/irbd" ]; then
    echo "ERROR: Output directory not found: data/preprocessed/irbd"
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
CONTROLS_COUNT=$(find data/raw/controls -name "*.cwa" -o -name "*.CWA" | wc -l)
IRBD_COUNT=$(find data/raw/irbd -name "*.cwa" -o -name "*.CWA" | wc -l)

echo "Input files found:"
echo "  - Controls: $CONTROLS_COUNT files"
echo "  - iRBD: $IRBD_COUNT files"
echo "  - Total: $((CONTROLS_COUNT + IRBD_COUNT)) files"

if [ $CONTROLS_COUNT -eq 0 ]; then
    echo "ERROR: No control .cwa files found"
    exit 1
fi

if [ $IRBD_COUNT -eq 0 ]; then
    echo "ERROR: No iRBD .cwa files found"
    exit 1
fi

# =============================================================================
# PREPROCESSING EXECUTION
# =============================================================================
echo "Starting preprocessing pipeline..."
echo "Timestamp: $(date)"

# Record start time
START_TIME=$(date +%s)

# Run preprocessing script
echo "Executing: python scripts/preprocessing.py"
python scripts/preprocessing.py

# Check exit status
PREPROCESSING_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Preprocessing completed with exit code: $PREPROCESSING_EXIT_CODE"
echo "Total processing time: $DURATION seconds ($(($DURATION / 60)) minutes)"

# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================
echo "Verifying preprocessing outputs..."

# Count output files
CONTROLS_OUTPUT=$(find data/preprocessed/controls -name "*.h5" | wc -l)
IRBD_OUTPUT=$(find data/preprocessed/irbd -name "*.h5" | wc -l)

echo "Output files created:"
echo "  - Controls: $CONTROLS_OUTPUT .h5 files"
echo "  - iRBD: $IRBD_OUTPUT .h5 files"
echo "  - Total: $((CONTROLS_OUTPUT + IRBD_OUTPUT)) .h5 files"

# Calculate success rate
TOTAL_INPUT=$((CONTROLS_COUNT + IRBD_COUNT))
TOTAL_OUTPUT=$((CONTROLS_OUTPUT + IRBD_OUTPUT))
SUCCESS_RATE=$(echo "scale=2; $TOTAL_OUTPUT * 100 / $TOTAL_INPUT" | bc)

echo "Processing success rate: $SUCCESS_RATE% ($TOTAL_OUTPUT/$TOTAL_INPUT)"

# =============================================================================
# QUALITY CHECKS
# =============================================================================
echo "Performing quality checks..."

# Check for log files
if [ -f "validation/data_quality_reports/preprocessing_"*".log" ]; then
    echo "✓ Preprocessing log file created"
    LOG_FILE=$(ls -t validation/data_quality_reports/preprocessing_*.log | head -1)
    echo "Latest log: $LOG_FILE"
    
    # Show last few lines of log
    echo "Last 10 lines of preprocessing log:"
    tail -10 "$LOG_FILE"
else
    echo "⚠ No preprocessing log file found"
fi

# Check for visualization outputs
if [ -d "results/visualizations" ] && [ "$(ls -A results/visualizations)" ]; then
    PLOT_COUNT=$(find results/visualizations -name "*.png" | wc -l)
    echo "✓ $PLOT_COUNT visualization plots created"
else
    echo "⚠ No visualization plots found"
fi

# Verify H5 file structure (sample check)
if [ $TOTAL_OUTPUT -gt 0 ]; then
    echo "Verifying H5 file structure..."
    SAMPLE_H5=$(find data/preprocessed -name "*.h5" | head -1)
    if [ -n "$SAMPLE_H5" ]; then
        echo "Sample file: $SAMPLE_H5"
        python -c "
import h5py
import sys
try:
    with h5py.File('$SAMPLE_H5', 'r') as f:
        print(f'✓ H5 file structure valid')
        print(f'  - Attributes: {list(f.attrs.keys())}')
        print(f'  - Datasets: {list(f.keys())}')
        if 'name' in f.attrs:
            print(f'  - Participant: {f.attrs[\"name\"]}')
        if 'number_of_nights' in f.attrs:
            print(f'  - Nights: {f.attrs[\"number_of_nights\"]}')
except Exception as e:
    print(f'✗ H5 file structure error: {e}')
    sys.exit(1)
"
    fi
fi

# =============================================================================
# CLEANUP AND SUMMARY
# =============================================================================
echo "Cleaning up temporary files..."

# Remove any temporary files (if created)
find /tmp -name "*actipy*" -user $(whoami) -delete 2>/dev/null || true

# =============================================================================
# JOB SUMMARY
# =============================================================================
echo "=========================================="
echo "PREPROCESSING JOB SUMMARY"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Exit Code: $PREPROCESSING_EXIT_CODE"
echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
echo "Input Files: $TOTAL_INPUT (.cwa files)"
echo "Output Files: $TOTAL_OUTPUT (.h5 files)"
echo "Success Rate: $SUCCESS_RATE%"
echo "Host: $LSB_HOSTS"
echo "Completed: $(date)"

# =============================================================================
# EXIT STATUS
# =============================================================================
if [ $PREPROCESSING_EXIT_CODE -eq 0 ] && [ $TOTAL_OUTPUT -gt 0 ]; then
    echo "✓ PREPROCESSING COMPLETED SUCCESSFULLY"
    echo "Ready for feature extraction stage"
    exit 0
else
    echo "✗ PREPROCESSING FAILED"
    echo "Check error logs and preprocessing output"
    exit 1
fi

