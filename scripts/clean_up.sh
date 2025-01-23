#!/bin/bash
# clean_up.sh: Script to clean up temporary files and optionally remove Conda environment.

ENV_NAME=${1:-"ml_env"}

echo "Cleaning up temporary files..."
rm -rf ./data/processed/*
rm -rf ./models/*
rm -rf ./results/*

read -p "Do you want to remove the Conda environment '$ENV_NAME'? (y/n) " -n 1 -r
echo    # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing Conda environment '$ENV_NAME'..."
    conda env remove -n $ENV_NAME
    echo "Environment '$ENV_NAME' removed."
fi

echo "Cleanup complete."