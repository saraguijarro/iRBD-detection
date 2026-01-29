#!/bin/bash
#BSUB -J gen_class_plots
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 0:30
#BSUB -o /work3/s184484/iRBD-detection/jobs/logs/plots/gen_class_plots_%J.out
#BSUB -e /work3/s184484/iRBD-detection/jobs/logs/plots/gen_class_plots_%J.err
#BSUB -N

# =============================================================================
# CLASSIFICATION PLOTS GENERATION JOB SCRIPT
# =============================================================================
# Purpose: Generate model performance comparison plots for thesis Results chapter
# Creates 5 comprehensive visualizations from LSTM and ML baseline results
# Versions: v0, v1, v1t, vvt (all preprocessing versions)
# =============================================================================

echo "=========================================="
echo "iRBD Detection Pipeline - Classification Plots"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Cores: $LSB_DJOB_NUMPROC"
echo "Host: $LSB_HOSTS"
echo "Started: $(date)"
echo ""
echo "Generating classification performance visualizations"
echo "Versions: v0, v1, v1t, vvt"
echo "=========================================="

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "Setting up environment..."

# Load conda module (DTU HPC specific)
module load miniconda3/4.12.0

# Activate visualization environment
echo "Activating conda environment: env_insights"
source activate env_insights

# Verify environment and key packages
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check critical packages
python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "ERROR: numpy not found"
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')" || echo "ERROR: matplotlib not found"
python -c "import seaborn; print(f'seaborn version: {seaborn.__version__}')" || echo "ERROR: seaborn not found"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
echo "Setting up directories..."

# Main project directory
PROJECT_DIR="/work3/s184484/iRBD-detection"
cd $PROJECT_DIR

# Script location
SCRIPT_PATH="$PROJECT_DIR/scripts/generate_classification_plots.py"

# Output directory (created by script, inside results/)
OUTPUT_DIR="$PROJECT_DIR/results/classification_plots"

echo "Project directory: $PROJECT_DIR"
echo "Script path: $SCRIPT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Working directory: $(pwd)"

# Verify script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "Script verified"

# Create logs directory if it doesn't exist
mkdir -p jobs/logs/plots

# =============================================================================
# INPUT VERIFICATION
# =============================================================================
echo ""
echo "=========================================="
echo "Verifying input files..."
echo "=========================================="

# Define all log file paths for all versions
declare -A LSTM_LOGS
declare -A ML_LOGS

LSTM_LOGS[v0]="$PROJECT_DIR/results/lstm_v0_night_level/training.log"
LSTM_LOGS[v1]="$PROJECT_DIR/results/lstm_v1_night_level/training.log"
LSTM_LOGS[v1t]="$PROJECT_DIR/results/lstm_v1t_night_level/training.log"
LSTM_LOGS[vvt]="$PROJECT_DIR/results/lstm_vvt_night_level/training.log"

ML_LOGS[v0]="$PROJECT_DIR/results/ml_baselines_v0/training.log"
ML_LOGS[v1]="$PROJECT_DIR/results/ml_baselines_v1/training.log"
ML_LOGS[v1t]="$PROJECT_DIR/results/ml_baselines_v1t/training.log"
ML_LOGS[vvt]="$PROJECT_DIR/results/ml_baselines_vvt/training.log"

echo "Checking for required log files:"
echo ""

MISSING=0
FOUND=0

for VERSION in v0 v1 v1t vvt; do
    echo "Version: $VERSION"
    
    # Check LSTM log
    LSTM_LOG="${LSTM_LOGS[$VERSION]}"
    if [ -f "$LSTM_LOG" ]; then
        SIZE=$(stat -c%s "$LSTM_LOG" 2>/dev/null || stat -f%z "$LSTM_LOG" 2>/dev/null)
        echo "  LSTM: Found (${SIZE} bytes)"
        FOUND=$((FOUND + 1))
    else
        echo "  LSTM: Missing - $LSTM_LOG"
        MISSING=$((MISSING + 1))
    fi
    
    # Check ML log
    ML_LOG="${ML_LOGS[$VERSION]}"
    if [ -f "$ML_LOG" ]; then
        SIZE=$(stat -c%s "$ML_LOG" 2>/dev/null || stat -f%z "$ML_LOG" 2>/dev/null)
        echo "  ML: Found (${SIZE} bytes)"
        FOUND=$((FOUND + 1))
    else
        echo "  ML: Missing - $ML_LOG"
        MISSING=$((MISSING + 1))
    fi
    echo ""
done

echo "Summary: $FOUND files found, $MISSING files missing"

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING log file(s) missing"
    echo "The script will continue but some plots may be incomplete"
    echo "Make sure LSTM and ML baseline training completed for all versions"
fi

# =============================================================================
# PLOT GENERATION
# =============================================================================
echo ""
echo "=========================================="
echo "Generating Classification Plots"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Run the plotting script
python "$SCRIPT_PATH"

# Check if script executed successfully
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Plot generation completed successfully"
    
    # List generated plots
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "Generated plots:"
        ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  (No PNG files found)"
        
        # Count plots
        PLOT_COUNT=$(ls -1 "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)
        echo ""
        echo "Total plots generated: $PLOT_COUNT"
    fi
else
    echo ""
    echo "ERROR: Plot generation failed with exit code $EXIT_CODE"
    exit 1
fi

# =============================================================================
# JOB COMPLETION
# =============================================================================
echo ""
echo "=========================================="
echo "JOB SUMMARY"
echo "=========================================="
echo "Versions processed: v0, v1, v1t, vvt"
echo "Input files found: $FOUND"
echo "Input files missing: $MISSING"
echo "Plots generated: $PLOT_COUNT"
echo "Output directory: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="

exit 0