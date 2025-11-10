#!/bin/bash

# DEPENDENCY MANAGEMENT AND MONITORING FOR iRBD DETECTION PIPELINE
# This script monitors job progress, handles failures, and provides status updates
# Can be run continuously to track pipeline completion

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
echo "=========================================="
echo "iRBD Detection Pipeline - Dependency Management"
echo "=========================================="

# Set script to continue on errors (we want to monitor and report)
set +e

# Configuration
PROJECT_DIR="/work3/s184484/irbd-detection"
STATUS_DIR="$PROJECT_DIR/jobs/status"
LOGS_DIR="$PROJECT_DIR/jobs/logs"
CHECK_INTERVAL=300  # Check every 5 minutes
MAX_CHECKS=288      # Maximum checks (24 hours with 5-minute intervals)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function to get current timestamp
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to log messages
log_message() {
    local message="$1"
    echo "[$(timestamp)] $message"
    echo "[$(timestamp)] $message" >> "$STATUS_DIR/pipeline_status.log"
}

# Function to get job status
get_job_status() {
    local job_id="$1"
    if [ -z "$job_id" ] || [ "$job_id" = "SKIPPED" ]; then
        echo "SKIPPED"
        return
    fi
    
    # Get job status from bjobs
    local status=$(bjobs -noheader "$job_id" 2>/dev/null | awk '{print $3}')
    
    if [ -z "$status" ]; then
        # Job not found in queue, check if it completed
        local exit_code=$(bjobs -a -noheader "$job_id" 2>/dev/null | awk '{print $3}')
        if [ "$exit_code" = "DONE" ]; then
            echo "DONE"
        elif [ "$exit_code" = "EXIT" ]; then
            echo "FAILED"
        else
            echo "UNKNOWN"
        fi
    else
        echo "$status"
    fi
}

# Function to get job runtime
get_job_runtime() {
    local job_id="$1"
    if [ -z "$job_id" ] || [ "$job_id" = "SKIPPED" ]; then
        echo "N/A"
        return
    fi
    
    local runtime=$(bjobs -l "$job_id" 2>/dev/null | grep -o 'Started on.*' | head -1)
    if [ -n "$runtime" ]; then
        echo "$runtime"
    else
        echo "Not started"
    fi
}

# Function to check stage completion by output files
check_stage_output() {
    local stage="$1"
    
    case $stage in
        "preprocessing")
            if [ -d "$PROJECT_DIR/data/preprocessed/controls" ] && [ -d "$PROJECT_DIR/data/preprocessed/irbd" ]; then
                local count=$(find "$PROJECT_DIR/data/preprocessed" -name "*.h5" | wc -l)
                if [ $count -gt 0 ]; then
                    echo "COMPLETE ($count files)"
                    return 0
                fi
            fi
            echo "INCOMPLETE"
            return 1
            ;;
        "feature_extraction")
            if [ -f "$PROJECT_DIR/data/features/combined/X_features.npy" ] && \
               [ -f "$PROJECT_DIR/data/features/combined/y_labels.npy" ]; then
                echo "COMPLETE"
                return 0
            fi
            echo "INCOMPLETE"
            return 1
            ;;
        "lstm")
            if [ -f "$PROJECT_DIR/results/lstm/models/best_lstm_model.pth" ] && \
               [ -f "$PROJECT_DIR/results/lstm/evaluation/evaluation_summary.json" ]; then
                echo "COMPLETE"
                return 0
            fi
            echo "INCOMPLETE"
            return 1
            ;;
        "statistical_analysis")
            if [ -f "$PROJECT_DIR/results/statistical_analysis/feature_analysis_summary.json" ]; then
                echo "COMPLETE"
                return 0
            fi
            echo "INCOMPLETE"
            return 1
            ;;
    esac
}

# Function to send notification (placeholder for email/slack integration)
send_notification() {
    local subject="$1"
    local message="$2"
    
    log_message "NOTIFICATION: $subject"
    log_message "$message"
    
    # TODO: Add email notification if needed
    # echo "$message" | mail -s "$subject" your.email@dtu.dk
}

# Function to display pipeline status
display_status() {
    echo ""
    echo "=========================================="
    echo "PIPELINE STATUS - $(timestamp)"
    echo "=========================================="
    
    # Load job IDs if available
    if [ -f "$STATUS_DIR/current_jobs.txt" ]; then
        source "$STATUS_DIR/current_jobs.txt"
    fi
    
    # Check each stage
    echo "Stage Status Summary:"
    echo ""
    
    # Preprocessing
    local prep_status=$(get_job_status "$PREPROCESSING")
    local prep_output=$(check_stage_output "preprocessing")
    printf "%-20s %-12s %-15s %s\n" "Preprocessing:" "$prep_status" "Job $PREPROCESSING" "$prep_output"
    
    # Feature Extraction
    local feat_status=$(get_job_status "$FEATURE_EXTRACTION")
    local feat_output=$(check_stage_output "feature_extraction")
    printf "%-20s %-12s %-15s %s\n" "Feature Extraction:" "$feat_status" "Job $FEATURE_EXTRACTION" "$feat_output"
    
    # LSTM Training
    local lstm_status=$(get_job_status "$LSTM_TRAINING")
    local lstm_output=$(check_stage_output "lstm")
    printf "%-20s %-12s %-15s %s\n" "LSTM Training:" "$lstm_status" "Job $LSTM_TRAINING" "$lstm_output"
    
    # Statistical Analysis
    local stats_status=$(get_job_status "$STATISTICAL_ANALYSIS")
    local stats_output=$(check_stage_output "statistical_analysis")
    printf "%-20s %-12s %-15s %s\n" "Statistical Analysis:" "$stats_status" "Job $STATISTICAL_ANALYSIS" "$stats_output"
    
    echo ""
    echo "Job Status Legend:"
    echo "  PEND    - Job pending in queue"
    echo "  RUN     - Job currently running"
    echo "  DONE    - Job completed successfully"
    echo "  FAILED  - Job failed"
    echo "  SKIPPED - Stage was skipped"
    echo ""
}

# Function to check for failed jobs and suggest actions
check_failures() {
    local failures=0
    
    if [ -f "$STATUS_DIR/current_jobs.txt" ]; then
        source "$STATUS_DIR/current_jobs.txt"
        
        # Check each job for failures
        for job_var in PREPROCESSING FEATURE_EXTRACTION LSTM_TRAINING STATISTICAL_ANALYSIS; do
            local job_id="${!job_var}"
            if [ -n "$job_id" ] && [ "$job_id" != "SKIPPED" ]; then
                local status=$(get_job_status "$job_id")
                if [ "$status" = "FAILED" ]; then
                    failures=$((failures + 1))
                    log_message "FAILURE DETECTED: $job_var (Job $job_id) failed"
                    
                    # Suggest checking error logs
                    case $job_var in
                        "PREPROCESSING")
                            log_message "Check error log: $LOGS_DIR/preprocessing/preprocessing_error_${job_id}.err"
                            ;;
                        "FEATURE_EXTRACTION")
                            log_message "Check error log: $LOGS_DIR/feature_extraction/feature_extraction_error_${job_id}.err"
                            ;;
                        "LSTM_TRAINING")
                            log_message "Check error log: $LOGS_DIR/lstm/lstm_error_${job_id}.err"
                            ;;
                        "STATISTICAL_ANALYSIS")
                            log_message "Check error log: $LOGS_DIR/stats/stats_error_${job_id}.err"
                            ;;
                    esac
                fi
            fi
        done
    fi
    
    return $failures
}

# Function to check pipeline completion
check_completion() {
    local all_complete=true
    
    # Check if all stages have completed successfully
    for stage in "preprocessing" "feature_extraction" "lstm" "statistical_analysis"; do
        local output_status=$(check_stage_output "$stage")
        if [[ ! "$output_status" =~ ^COMPLETE ]]; then
            all_complete=false
            break
        fi
    done
    
    if [ "$all_complete" = true ]; then
        log_message "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
        log_message "All stages have completed and output files are present"
        
        # Generate completion summary
        generate_completion_summary
        
        send_notification "iRBD Pipeline Completed" "All pipeline stages completed successfully. Results are ready for analysis."
        return 0
    fi
    
    return 1
}

# Function to generate completion summary
generate_completion_summary() {
    local summary_file="$STATUS_DIR/pipeline_completion_summary.txt"
    
    echo "iRBD Detection Pipeline - Completion Summary" > "$summary_file"
    echo "Generated: $(timestamp)" >> "$summary_file"
    echo "=========================================" >> "$summary_file"
    echo "" >> "$summary_file"
    
    # Count output files
    local h5_count=$(find "$PROJECT_DIR/data/preprocessed" -name "*.h5" 2>/dev/null | wc -l)
    local npy_count=$(find "$PROJECT_DIR/data/features" -name "*_features.npy" 2>/dev/null | wc -l)
    
    echo "Output Summary:" >> "$summary_file"
    echo "  - Preprocessed files: $h5_count .h5 files" >> "$summary_file"
    echo "  - Feature files: $npy_count .npy files" >> "$summary_file"
    
    if [ -f "$PROJECT_DIR/data/features/combined/dataset_info.json" ]; then
        echo "  - Combined dataset: Available" >> "$summary_file"
    fi
    
    if [ -f "$PROJECT_DIR/results/lstm/evaluation/evaluation_summary.json" ]; then
        echo "  - LSTM model: Trained and evaluated" >> "$summary_file"
    fi
    
    if [ -f "$PROJECT_DIR/results/statistical_analysis/feature_analysis_summary.json" ]; then
        echo "  - Statistical analysis: Complete" >> "$summary_file"
    fi
    
    echo "" >> "$summary_file"
    echo "Key Results Locations:" >> "$summary_file"
    echo "  - LSTM Results: $PROJECT_DIR/results/lstm/" >> "$summary_file"
    echo "  - Statistical Results: $PROJECT_DIR/results/statistical_analysis/" >> "$summary_file"
    echo "  - Visualizations: $PROJECT_DIR/results/visualizations/" >> "$summary_file"
    echo "" >> "$summary_file"
    
    log_message "Completion summary saved to: $summary_file"
}

# =============================================================================
# MAIN MONITORING LOOP
# =============================================================================

# Check if we're in the correct directory
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# Verify status directory exists (should already exist)
if [ ! -d "$STATUS_DIR" ]; then
    echo "ERROR: Status directory not found: $STATUS_DIR"
    echo "Make sure the complete project structure is set up"
    exit 1
fi

# Check if job IDs file exists
if [ ! -f "$STATUS_DIR/current_jobs.txt" ]; then
    echo "WARNING: No current jobs file found at $STATUS_DIR/current_jobs.txt"
    echo "This script monitors jobs submitted via submit_pipeline.sh"
    echo ""
    echo "You can still use this script to check pipeline status manually."
    echo "Press Ctrl+C to exit, or Enter to continue with manual monitoring..."
    read
fi

log_message "Starting pipeline monitoring..."
log_message "Project directory: $PROJECT_DIR"
log_message "Check interval: $CHECK_INTERVAL seconds"

# Main monitoring loop
check_count=0
while [ $check_count -lt $MAX_CHECKS ]; do
    # Display current status
    display_status
    
    # Check for failures
    check_failures
    failure_count=$?
    
    # Check for completion
    if check_completion; then
        log_message "Pipeline monitoring completed - all stages finished successfully"
        exit 0
    fi
    
    # If there are failures, ask user what to do
    if [ $failure_count -gt 0 ]; then
        echo ""
        echo "⚠ $failure_count job(s) failed. Options:"
        echo "1. Continue monitoring (ignore failures)"
        echo "2. Exit monitoring"
        echo "3. Show detailed failure information"
        echo ""
        read -p "Choose option (1-3) or press Enter to continue: " -n 1 -r
        echo ""
        
        case $REPLY in
            2)
                log_message "Monitoring stopped by user due to failures"
                exit 1
                ;;
            3)
                echo ""
                echo "Detailed failure information:"
                bjobs -a | grep -E "(FAILED|EXIT)"
                echo ""
                echo "Check error logs in $LOGS_DIR/"
                echo ""
                ;;
        esac
    fi
    
    # Wait before next check
    if [ $check_count -lt $((MAX_CHECKS - 1)) ]; then
        echo ""
        echo "Next check in $CHECK_INTERVAL seconds... (Press Ctrl+C to stop monitoring)"
        echo "Check $((check_count + 1))/$MAX_CHECKS"
        sleep $CHECK_INTERVAL
    fi
    
    check_count=$((check_count + 1))
done

# If we reach here, monitoring timed out
log_message "Monitoring timed out after $MAX_CHECKS checks ($(($MAX_CHECKS * $CHECK_INTERVAL / 3600)) hours)"
log_message "Pipeline may still be running. Check job status manually with 'bjobs'"

echo ""
echo "=========================================="
echo "MONITORING TIMEOUT"
echo "=========================================="
echo "Monitoring stopped after $((MAX_CHECKS * CHECK_INTERVAL / 3600)) hours"
echo "Pipeline may still be running."
echo ""
echo "To continue monitoring:"
echo "  bash jobs/scripts/dependency_management.sh"
echo ""
echo "To check status manually:"
echo "  bjobs                    # Check job status"
echo "  bjobs -u s184484         # Check your jobs"
echo ""

exit 2

