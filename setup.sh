#!/bin/bash
set -e

echo "Setting up VigilEye environment..."

# Use Python 3.12 if available, otherwise try to find a suitable python3
PYTHON_EXE=$(which python3.12 || which python3)

if [[ ! -d ".py312" ]]; then
    echo "Creating virtual environment in .py312..."
    $PYTHON_EXE -m venv .py312
fi

echo "Installing dependencies..."
.py312/bin/python -m pip install --upgrade pip
.py312/bin/python -m pip install .

echo "Environment setup complete. Use ./.py312/bin/python to run VigilEye."
