#!/bin/bash
# evaluate_model.sh: Script to evaluate a trained model.

ENV_NAME=${1:-"ml_env"}
MODEL_PATH=${2:-"./models/model.pth"}
TEST_DATASET=${3:-"./data/test.csv"}
METRICS_OUTPUT=${4:-"./results/metrics.json"}

echo "Activating Conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Evaluating model..."
python evaluate.py \
  --model_path $MODEL_PATH \
  --test_dataset $TEST_DATASET \
  --output_metrics $METRICS_OUTPUT

echo "Evaluation complete. Metrics saved in $METRICS_OUTPUT."