#!/bin/bash
#BSUB -J cwa_investigate
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o /work3/s184484/iRBD-detection/jobs/logs/cwa_investigate/cwa_investigate_%J.out
#BSUB -e /work3/s184484/iRBD-detection/jobs/logs/cwa_investigate/cwa_investigate_%J.err

# Job script to investigate all .cwa files in the dataset
# This runs the investigation as a batch job to avoid timeout issues

echo "=========================================="
echo "CWA File Investigation Job"
echo "Job ID: $LSB_JOBID"
echo "Started: $(date)"
echo "=========================================="

# Load environment
source activate env_preprocessing


# Set working directory
cd /work3/s184484/iRBD-detection/scripts

# Run the investigation
echo "Starting investigation of all CWA files..."
python investigate_cwa_structure.py \
    /work3/s184484/iRBD-detection/data/raw/ \
    --output /work3/s184484/iRBD-detection/results/full_cwa_metadata.json

EXIT_CODE=$?

echo "=========================================="
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Investigation completed successfully"
    echo "Results saved to: /work3/s184484/iRBD-detection/results/full_cwa_metadata.json"
else
    echo "ERROR: Investigation failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
