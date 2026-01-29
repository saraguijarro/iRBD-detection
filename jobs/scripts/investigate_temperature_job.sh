#!/bin/bash
#BSUB -J temp_invest
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /work3/s184484/iRBD-detection/jobs/logs/temp_investigation/temp_invest_output_%J.out
#BSUB -e /work3/s184484/iRBD-detection/jobs/logs/temp_investigation/temp_invest_error_%J.err
#BSUB -N

# =============================================================================
# TEMPERATURE INVESTIGATION - HPC JOB SCRIPT
# =============================================================================
# Purpose: Run comprehensive temperature analysis on raw .cwa files
# =============================================================================

echo "=========================================="
echo "TEMPERATURE INVESTIGATION"
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

# Activate preprocessing environment (conda)
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
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')" || echo "WARNING: matplotlib not found"
python -c "import seaborn; print(f'seaborn version: {seaborn.__version__}')" || echo "WARNING: seaborn not found"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
echo "Setting up directories..."

# Main project directory
PROJECT_DIR="/work3/s184484/iRBD-detection"
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

# Create output directory if it doesn't exist
echo "Creating output directories..."
mkdir -p results/temperature_investigation
mkdir -p jobs/logs/temp_investigation

echo "Output directory: results/temperature_investigation"
echo "Project directory: $PROJECT_DIR"
echo "Working directory: $(pwd)"
echo "All required directories verified"

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
# TEMPERATURE INVESTIGATION EXECUTION
# =============================================================================
echo "Starting temperature investigation..."
echo "Timestamp: $(date)"

# Record start time
START_TIME=$(date +%s)

# Run temperature investigation script
echo "Executing: python scripts/investigate_temperature.py"
python scripts/investigate_temperature.py

# Check exit status
SCRIPT_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Temperature investigation completed with exit code: $SCRIPT_EXIT_CODE"
echo "Total processing time: $DURATION seconds ($(($DURATION / 60)) minutes)"

# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================
echo "Verifying outputs..."

# Count output files
PNG_COUNT=$(find results/temperature_investigation -name "*.png" 2>/dev/null | wc -l)
CSV_COUNT=$(find results/temperature_investigation -name "*.csv" 2>/dev/null | wc -l)
TXT_COUNT=$(find results/temperature_investigation -name "*.txt" 2>/dev/null | wc -l)

echo "Output files created:"
echo "  - PNG figures: $PNG_COUNT"
echo "  - CSV data files: $CSV_COUNT"
echo "  - TXT reports: $TXT_COUNT"

# List all output files
echo ""
echo "Generated files:"
ls -la results/temperature_investigation/

# =============================================================================
# JOB SUMMARY
# =============================================================================
echo "=========================================="
echo "TEMPERATURE INVESTIGATION JOB SUMMARY"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Exit Code: $SCRIPT_EXIT_CODE"
echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
echo "Input Files: $((CONTROLS_COUNT + IRBD_COUNT)) (.cwa files)"
echo "Output PNG: $PNG_COUNT"
echo "Output CSV: $CSV_COUNT"
echo "Host: $LSB_HOSTS"
echo "Completed: $(date)"

# =============================================================================
# EXIT STATUS
# =============================================================================
if [ $SCRIPT_EXIT_CODE -eq 0 ] && [ $PNG_COUNT -gt 0 ]; then
    echo "TEMPERATURE INVESTIGATION COMPLETED SUCCESSFULLY"
    exit 0
else
    echo "TEMPERATURE INVESTIGATION FAILED"
    echo "Check error logs for details"
    exit 1
fi