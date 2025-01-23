#!/bin/bash
# run_pipeline.sh: Script to run the entire ML pipeline.

ENV_NAME=${1:-"ml_env"}
RAW_DATA_PATH=${2:-"./data/raw"}
PROCESSED_DATA_PATH=${3:-"./data/processed"}
MODEL_DIR=${4:-"./models"}
RESULTS_DIR=${5:-"./results"}
EPOCHS=${6:-10}
BATCH_SIZE=${7:-32}

echo "Starting ML pipeline..."

# Step 1: Preprocess data
echo "Preprocessing data..."
bash preprocess_data.sh $ENV_NAME $RAW_DATA_PATH $PROCESSED_DATA_PATH

# Step 2: Train model
echo "Training model..."
bash train_model.sh $ENV_NAME $PROCESSED_DATA_PATH/train.csv $MODEL_DIR $EPOCHS $BATCH_SIZE

# Step 3: Evaluate model
echo "Evaluating model..."
bash evaluate_model.sh $ENV_NAME $MODEL_DIR/model.pth $PROCESSED_DATA_PATH/test.csv $RESULTS_DIR/metrics.json

echo "ML pipeline complete. Results saved in $RESULTS_DIR."