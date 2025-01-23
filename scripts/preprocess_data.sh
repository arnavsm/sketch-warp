#!/bin/bash
# preprocess_data.sh: Script to preprocess raw data.

ENV_NAME=${1:-"ml_env"}
RAW_DATA_PATH=${2:-"./data/raw"}
PROCESSED_DATA_PATH=${3:-"./data/processed"}

echo "Activating Conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Starting data preprocessing..."
python preprocess.py \
  --input_dir $RAW_DATA_PATH \
  --output_dir $PROCESSED_DATA_PATH

echo "Data preprocessing complete. Processed data saved in $PROCESSED_DATA_PATH."