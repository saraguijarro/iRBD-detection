#!/bin/bash
#BSUB -J feature_v1t
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o /work3/s184484/iRBD-detection/jobs/logs/feature_v1t/feature_v1t_%J.out
#BSUB -e /work3/s184484/iRBD-detection/jobs/logs/feature_v1t/feature_v1t_%J.err
#BSUB -N

# Feature Extraction V1T - Per-sample temperature filtering (20°C threshold)
# Input: /work3/s184484/iRBD-detection/data/preprocessed_v1t/
# Output: /work3/s184484/iRBD-detection/data/features_v1t/

echo "========================================="
echo "Feature Extraction V1T Job Started"
echo "Job ID: $LSB_JOBID"
echo "Started at: $(date)"
echo "========================================="

# Load required modules
module load python3/3.11.13
module load cuda/12.1

# Activate conda environment (DTU HPC method)
echo "Activating conda environment: env_insights"
source activate env_insights

# Print environment info
echo "Python version: $(python --version)"
echo "CUDA available:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

# Navigate to project directory
cd /work3/s184484/iRBD-detection

# Create log and output directories if they don't exist
mkdir -p jobs/logs/feature_v1t
mkdir -p data/features_v1t/controls
mkdir -p data/features_v1t/irbd

echo "Input directory: data/preprocessed_v1t/"
echo "Output directory: data/features_v1t/"
echo ""

# Run unified feature extraction with v1t argument
echo "Starting feature extraction..."
python scripts/feature_extraction.py --version v1t

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Feature Extraction V1T Job Completed"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE
