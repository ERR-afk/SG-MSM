#!/usr/bin/env bash

# Weights & Biases Configuration
export WANDB_API_KEY="xxxxxxxx"
export WANDB_MODE="online" 

export HF_HUB_OFFLINE=0

# Ensure the correct conda environment is active before running
# conda activate sgmsm_env

cd "$(dirname "$0")"

CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Please provide a config file."
    echo "Usage: ./check_and_launch.sh configs/config_1_perch_no_sgmsm.json"
    exit 1
fi

echo "Launching Birdset Multi-Domain Training with config: $CONFIG_PATH"
python run_training.py --cfg $CONFIG_PATH