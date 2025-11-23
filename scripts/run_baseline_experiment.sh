#!/usr/bin/env zsh
# Convenience script for running baseline prompting experiments

set -e

# Get the directory where this script is located
SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"

# Default config
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/baseline_experiment.yaml}"

echo "=== SDF CoT Monitorability: Baseline Experiment ==="
echo "Project root: $PROJECT_ROOT"
echo "Config: $CONFIG"
echo ""

# Check if .env exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Warning: .env file not found. Creating from .env.example..."
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo "Please edit .env with your API keys before running."
        exit 1
    else
        echo "Error: .env.example not found"
        exit 1
    fi
fi

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Run the experiment
cd "$PROJECT_ROOT"
python scripts/run_baseline_experiment.py --config "$CONFIG" "$@"

