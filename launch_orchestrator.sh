#!/bin/bash
# Launch script for the Experiment Orchestrator

# Activate the virtual environment
source /home/maximilienleclei/venvs/orchestator/bin/activate

# Run the orchestrator
python -m orchestrator.main
