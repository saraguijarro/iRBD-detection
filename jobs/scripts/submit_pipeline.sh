#!/bin/bash

# AUTOMATED PIPELINE SUBMISSION FOR iRBD DETECTION
# This script submits all pipeline jobs with proper dependencies
# Ensures each stage completes successfully before starting the next

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
echo "=========================================="
echo "iRBD Detection Pipeline - Automated Submission"
echo "=========================================="
echo "This script will submit all pipeline jobs with dependencies:"
echo "1. Preprocessing (raw .cwa → clean .h5)"
echo "2. Feature Extraction (clean .h5 → SSL-Wearables features)"
echo "3. LSTM Training (features → classification model)"
echo "4. Statistical Analysis (features → group comparisons)"
echo "=========================================="

# Set script to exit on any error
set -e

# Record start time
START_TIME=$(date +%s)

# =============================================================================
# DIRECTORY AND FILE VERIFICATION
# =============================================================================
echo "Verifying pipeline setup..."

# Check if we're in the correct directory
PROJECT_DIR="/work3/s184484/iRBD-detection"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    echo "Make sure you're running this from the correct location"
    exit 1
fi

cd "$PROJECT_DIR"
echo "✓ Project directory: $PROJECT_DIR"

# Verify job scripts exist
JOB_SCRIPTS=(
    "jobs/scripts/preprocessing_job.sh"
    "jobs/scripts/feature_extraction_job.sh"
    "jobs/scripts/lstm_job.sh"
    "jobs/scripts/statistical_analysis_job.sh"
)

MISSING_SCRIPTS=0
for script in "${JOB_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "✓ Found: $script"
    else
        echo "✗ Missing: $script"
        MISSING_SCRIPTS=$((MISSING_SCRIPTS + 1))
    fi
done

if [ $MISSING_SCRIPTS -gt 0 ]; then
    echo "ERROR: $MISSING_SCRIPTS job scripts missing"
    echo "Make sure all job scripts are in jobs/scripts/ directory"
    exit 1
fi

# Verify Python scripts exist
PYTHON_SCRIPTS=(
    "scripts/preprocessing.py"
    "scripts/feature_extraction.py"
    "scripts/lstm.py"
    "scripts/statistical_analysis.py"
)

MISSING_PYTHON=0
for script in "${PYTHON_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "✓ Found: $script"
    else
        echo "✗ Missing: $script"
        MISSING_PYTHON=$((MISSING_PYTHON + 1))
    fi
done

if [ $MISSING_PYTHON -gt 0 ]; then
    echo "ERROR: $MISSING_PYTHON Python scripts missing"
    echo "Make sure all Python scripts are in scripts/ directory"
    exit 1
fi

# Verify input data exists
if [ ! -d "data/raw/controls" ] || [ ! -d "data/raw/irbd" ]; then
    echo "ERROR: Raw data directories not found"
    echo "Make sure data/raw/controls/ and data/raw/irbd/ exist with .cwa files"
    exit 1
fi

CONTROLS_COUNT=$(find data/raw/controls -name "*.cwa" -o -name "*.CWA" | wc -l)
IRBD_COUNT=$(find data/raw/irbd -name "*.cwa" -o -name "*.CWA" | wc -l)

echo "✓ Input data found:"
echo "  - Controls: $CONTROLS_COUNT .cwa files"
echo "  - iRBD: $IRBD_COUNT .cwa files"

if [ $CONTROLS_COUNT -eq 0 ] || [ $IRBD_COUNT -eq 0 ]; then
    echo "ERROR: Insufficient input data"
    exit 1
fi

# =============================================================================
# DIRECTORY STRUCTURE VERIFICATION
# =============================================================================
echo "Verifying directory structure..."

# Check that all required directories exist (they should already be created)
REQUIRED_DIRS=(
    "data/preprocessed/controls"
    "data/preprocessed/irbd"
    "data/features/controls"
    "data/features/irbd"
    "data/features/combined"
    "results/lstm/models"
    "results/lstm/predictions"
    "results/lstm/evaluation"
    "results/lstm/interpretability"
    "results/statistical_analysis"
    "results/visualizations"
    "validation/data_quality_reports"
    "jobs/logs/preprocessing"
    "jobs/logs/feature_extraction"
    "jobs/logs/lstm"
    "jobs/logs/stats"
    "jobs/status"
)

MISSING_DIRS=0
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ Directory exists: $dir"
    else
        echo "✗ Missing directory: $dir"
        MISSING_DIRS=$((MISSING_DIRS + 1))
    fi
done

if [ $MISSING_DIRS -gt 0 ]; then
    echo "ERROR: $MISSING_DIRS required directories missing"
    echo "Make sure the complete project structure is set up"
    echo "All directories should be created before running the pipeline"
    exit 1
fi

echo "✓ All required directories verified"

# =============================================================================
# ENVIRONMENT VERIFICATION
# =============================================================================
echo ""
echo "Verifying conda environments..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Load conda module first:"
    echo "module load miniconda3/4.12.0"
    exit 1
fi

# Check if required environments exist
if ! conda env list | grep -q "env_preprocessing"; then
    echo "ERROR: env_preprocessing environment not found"
    echo "Create it first: conda env create -f environments/env_preprocessing.yml"
    exit 1
fi

if ! conda env list | grep -q "env_insights"; then
    echo "ERROR: env_insights environment not found"
    echo "Create it first: conda env create -f environments/env_insights.yml"
    exit 1
fi

echo "✓ Required conda environments found"

# =============================================================================
# PIPELINE SUBMISSION OPTIONS
# =============================================================================
echo ""
echo "Pipeline submission options:"
echo "1. Submit complete pipeline (recommended)"
echo "2. Submit individual stages"
echo "3. Resume from specific stage"
echo ""

read -p "Choose option (1-3): " -n 1 -r OPTION
echo ""

case $OPTION in
    1)
        echo "Submitting complete pipeline with dependencies..."
        SUBMIT_COMPLETE=true
        ;;
    2)
        echo "Individual stage submission selected"
        SUBMIT_COMPLETE=false
        ;;
    3)
        echo "Resume pipeline selected"
        SUBMIT_COMPLETE=false
        echo "Available stages:"
        echo "1. Preprocessing"
        echo "2. Feature Extraction"
        echo "3. LSTM Training"
        echo "4. Statistical Analysis"
        read -p "Resume from stage (1-4): " -n 1 -r RESUME_STAGE
        echo ""
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

# =============================================================================
# JOB SUBMISSION FUNCTIONS
# =============================================================================

# Function to submit a job and extract job ID
submit_job() {
    local job_script=$1
    local job_name=$2
    local dependency=$3
    
    echo "Submitting $job_name..."
    
    if [ -n "$dependency" ]; then
        echo "  - Dependency: Job $dependency must complete successfully"
        JOB_OUTPUT=$(bsub -w "done($dependency)" < "$job_script")
    else
        echo "  - No dependencies"
        JOB_OUTPUT=$(bsub < "$job_script")
    fi
    
    # Extract job ID from bsub output
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*')
    
    if [ -n "$JOB_ID" ]; then
        echo "  - ✓ Submitted successfully: Job ID $JOB_ID"
        echo "$JOB_OUTPUT"
    else
        echo "  - ✗ Submission failed"
        echo "$JOB_OUTPUT"
        exit 1
    fi
    
    echo "$JOB_ID"
}

# Function to check if a stage has already completed
check_stage_completion() {
    local stage=$1
    
    case $stage in
        "preprocessing")
            if [ -d "data/preprocessed/controls" ] && [ -d "data/preprocessed/irbd" ]; then
                PREP_COUNT=$(find data/preprocessed -name "*.h5" | wc -l)
                if [ $PREP_COUNT -gt 0 ]; then
                    echo "✓ Preprocessing appears complete ($PREP_COUNT .h5 files found)"
                    return 0
                fi
            fi
            ;;
        "feature_extraction")
            if [ -f "data/features/combined/X_features.npy" ] && [ -f "data/features/combined/y_labels.npy" ]; then
                echo "✓ Feature extraction appears complete"
                return 0
            fi
            ;;
        "lstm")
            if [ -f "results/lstm/models/best_lstm_model.pth" ]; then
                echo "✓ LSTM training appears complete"
                return 0
            fi
            ;;
        "statistical_analysis")
            if [ -f "results/statistical_analysis/feature_analysis_summary.json" ]; then
                echo "✓ Statistical analysis appears complete"
                return 0
            fi
            ;;
    esac
    
    return 1
}

# =============================================================================
# COMPLETE PIPELINE SUBMISSION
# =============================================================================

if [ "$SUBMIT_COMPLETE" = true ]; then
    echo ""
    echo "=========================================="
    echo "SUBMITTING COMPLETE PIPELINE"
    echo "=========================================="
    
    # Check for existing completions
    echo "Checking for already completed stages..."
    
    # Submit preprocessing
    if check_stage_completion "preprocessing"; then
        read -p "Preprocessing appears complete. Skip? (Y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "Skipping preprocessing..."
            PREP_JOB_ID="SKIPPED"
        else
            PREP_JOB_ID=$(submit_job "jobs/scripts/preprocessing_job.sh" "Preprocessing" "")
        fi
    else
        PREP_JOB_ID=$(submit_job "jobs/scripts/preprocessing_job.sh" "Preprocessing" "")
    fi
    
    # Submit feature extraction (depends on preprocessing)
    if [ "$PREP_JOB_ID" = "SKIPPED" ] && check_stage_completion "feature_extraction"; then
        read -p "Feature extraction appears complete. Skip? (Y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "Skipping feature extraction..."
            FEAT_JOB_ID="SKIPPED"
        else
            FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "")
        fi
    else
        if [ "$PREP_JOB_ID" = "SKIPPED" ]; then
            FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "")
        else
            FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "$PREP_JOB_ID")
        fi
    fi
    
    # Submit LSTM training (depends on feature extraction)
    if [ "$FEAT_JOB_ID" = "SKIPPED" ]; then
        LSTM_JOB_ID=$(submit_job "jobs/scripts/lstm_job.sh" "LSTM Training" "")
    else
        LSTM_JOB_ID=$(submit_job "jobs/scripts/lstm_job.sh" "LSTM Training" "$FEAT_JOB_ID")
    fi
    
    # Submit statistical analysis (depends on feature extraction, can run parallel with LSTM)
    if [ "$FEAT_JOB_ID" = "SKIPPED" ]; then
        STATS_JOB_ID=$(submit_job "jobs/scripts/statistical_analysis_job.sh" "Statistical Analysis" "")
    else
        STATS_JOB_ID=$(submit_job "jobs/scripts/statistical_analysis_job.sh" "Statistical Analysis" "$FEAT_JOB_ID")
    fi
    
    # =============================================================================
    # PIPELINE SUMMARY
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "PIPELINE SUBMISSION SUMMARY"
    echo "=========================================="
    echo "Submitted jobs:"
    [ "$PREP_JOB_ID" != "SKIPPED" ] && echo "  - Preprocessing: Job $PREP_JOB_ID"
    [ "$FEAT_JOB_ID" != "SKIPPED" ] && echo "  - Feature Extraction: Job $FEAT_JOB_ID"
    echo "  - LSTM Training: Job $LSTM_JOB_ID"
    echo "  - Statistical Analysis: Job $STATS_JOB_ID"
    echo ""
    echo "Job dependencies:"
    if [ "$PREP_JOB_ID" != "SKIPPED" ] && [ "$FEAT_JOB_ID" != "SKIPPED" ]; then
        echo "  - Feature Extraction waits for Preprocessing ($PREP_JOB_ID)"
    fi
    if [ "$FEAT_JOB_ID" != "SKIPPED" ]; then
        echo "  - LSTM Training waits for Feature Extraction ($FEAT_JOB_ID)"
        echo "  - Statistical Analysis waits for Feature Extraction ($FEAT_JOB_ID)"
    fi
    echo ""
    echo "Monitor progress with:"
    echo "  bjobs                    # Check job status"
    echo "  bjobs -l <job_id>        # Detailed job info"
    echo "  bash jobs/scripts/dependency_management.sh  # Automated monitoring"
    
    # Save job IDs for monitoring
    echo "# iRBD Detection Pipeline Job IDs - $(date)" > jobs/status/current_jobs.txt
    [ "$PREP_JOB_ID" != "SKIPPED" ] && echo "PREPROCESSING=$PREP_JOB_ID" >> jobs/status/current_jobs.txt
    [ "$FEAT_JOB_ID" != "SKIPPED" ] && echo "FEATURE_EXTRACTION=$FEAT_JOB_ID" >> jobs/status/current_jobs.txt
    echo "LSTM_TRAINING=$LSTM_JOB_ID" >> jobs/status/current_jobs.txt
    echo "STATISTICAL_ANALYSIS=$STATS_JOB_ID" >> jobs/status/current_jobs.txt
    
    echo ""
    echo "✓ Complete pipeline submitted successfully!"
    echo "Job IDs saved to: jobs/status/current_jobs.txt"
    
fi

# =============================================================================
# INDIVIDUAL STAGE SUBMISSION
# =============================================================================

if [ "$SUBMIT_COMPLETE" = false ] && [ -z "$RESUME_STAGE" ]; then
    echo ""
    echo "Individual stage submission - select stages to submit:"
    echo ""
    
    read -p "Submit preprocessing? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        PREP_JOB_ID=$(submit_job "jobs/scripts/preprocessing_job.sh" "Preprocessing" "")
    fi
    
    read -p "Submit feature extraction? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -n "$PREP_JOB_ID" ]; then
            read -p "Wait for preprocessing to complete? (Y/n): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "$PREP_JOB_ID")
            else
                FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "")
            fi
        else
            FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "")
        fi
    fi
    
    read -p "Submit LSTM training? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -n "$FEAT_JOB_ID" ]; then
            LSTM_JOB_ID=$(submit_job "jobs/scripts/lstm_job.sh" "LSTM Training" "$FEAT_JOB_ID")
        else
            LSTM_JOB_ID=$(submit_job "jobs/scripts/lstm_job.sh" "LSTM Training" "")
        fi
    fi
    
    read -p "Submit statistical analysis? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -n "$FEAT_JOB_ID" ]; then
            STATS_JOB_ID=$(submit_job "jobs/scripts/statistical_analysis_job.sh" "Statistical Analysis" "$FEAT_JOB_ID")
        else
            STATS_JOB_ID=$(submit_job "jobs/scripts/statistical_analysis_job.sh" "Statistical Analysis" "")
        fi
    fi
fi

# =============================================================================
# RESUME PIPELINE SUBMISSION
# =============================================================================

if [ -n "$RESUME_STAGE" ]; then
    echo ""
    echo "Resuming pipeline from stage $RESUME_STAGE..."
    
    case $RESUME_STAGE in
        1)
            echo "Resuming from preprocessing..."
            PREP_JOB_ID=$(submit_job "jobs/scripts/preprocessing_job.sh" "Preprocessing" "")
            FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "$PREP_JOB_ID")
            LSTM_JOB_ID=$(submit_job "jobs/scripts/lstm_job.sh" "LSTM Training" "$FEAT_JOB_ID")
            STATS_JOB_ID=$(submit_job "jobs/scripts/statistical_analysis_job.sh" "Statistical Analysis" "$FEAT_JOB_ID")
            ;;
        2)
            echo "Resuming from feature extraction..."
            FEAT_JOB_ID=$(submit_job "jobs/scripts/feature_extraction_job.sh" "Feature Extraction" "")
            LSTM_JOB_ID=$(submit_job "jobs/scripts/lstm_job.sh" "LSTM Training" "$FEAT_JOB_ID")
            STATS_JOB_ID=$(submit_job "jobs/scripts/statistical_analysis_job.sh" "Statistical Analysis" "$FEAT_JOB_ID")
            ;;
        3)
            echo "Resuming from LSTM training..."
            LSTM_JOB_ID=$(submit_job "jobs/scripts/lstm_job.sh" "LSTM Training" "")
            ;;
        4)
            echo "Resuming from statistical analysis..."
            STATS_JOB_ID=$(submit_job "jobs/scripts/statistical_analysis_job.sh" "Statistical Analysis" "")
            ;;
    esac
fi

# =============================================================================
# FINAL INSTRUCTIONS
# =============================================================================
echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Monitor job progress:"
echo "   bjobs                    # Check all your jobs"
echo "   bjobs -u s184484         # Check your specific jobs"
echo "   bjobs -l <job_id>        # Detailed job information"
echo ""
echo "2. Check job outputs:"
echo "   tail -f jobs/logs/*/output_<job_id>.out    # Follow job output"
echo "   tail -f jobs/logs/*/error_<job_id>.err     # Check for errors"
echo ""
echo "3. Use automated monitoring:"
echo "   bash jobs/scripts/dependency_management.sh  # Monitor pipeline progress"
echo ""
echo "4. Pipeline completion:"
echo "   - Preprocessing: ~2-6 hours"
echo "   - Feature Extraction: ~4-12 hours"
echo "   - LSTM Training: ~6-24 hours"
echo "   - Statistical Analysis: ~1-3 hours"
echo "   - Total pipeline: ~13-45 hours"
echo ""

# Calculate submission time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Pipeline submission completed in $DURATION seconds"
echo "Started: $(date -d @$START_TIME)"
echo "Completed: $(date)"
echo ""
echo "✓ iRBD Detection Pipeline submitted successfully!"

