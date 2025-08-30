#!/bin/bash
# setup.sh: Script to set up Conda environment and install dependencies.

ENV_NAME=${1:-"warp_env"}  # Default environment name is 'warp_env'

echo "Creating Conda environment '$ENV_NAME' from environment.yml..."
conda env create -f environment.yml -n $ENV_NAME

echo "Activating Conda environment..."
conda activate $ENV_NAME

echo "Environment setup complete."