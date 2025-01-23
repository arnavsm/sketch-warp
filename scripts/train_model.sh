#!/bin/bash
# train_model.sh: Script to train a machine learning model.

ENV_NAME=${1:-"ml_env"}
DATASET_PATH=${2:-"./data/train.csv"}
MODEL_DIR=${3:-"./models"}
EPOCHS=${4:-10}
BATCH_SIZE=${5:-32}

echo "Activating Conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Starting training..."
python train.py \
  --dataset $DATASET_PATH \
  --model_dir $MODEL_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE

echo "Training complete. Model saved in $MODEL_DIR."