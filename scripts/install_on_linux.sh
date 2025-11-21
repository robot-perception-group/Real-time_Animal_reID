#!/usr/bin/env bash

# ---------------------------------------------------
# RAPID Linux setup script with error handling
# ---------------------------------------------------

# Step 0: Go to the directory where this script is located
cd "$(dirname "$0")" || { echo "Failed to enter install script's directory"; exit 1; }

# Step 1: Move one directory up (parent folder)
cd .. || { echo "Failed to move to project root"; exit 1; }

# Step 1: Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv || { echo "Failed to create virtual environment. Do you have Python3 installed?"; exit 1; }

# Step 2: Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Step 3: Install uv
echo "Installing uv..."
pip install uv || { echo "Failed to install uv. Check your internet connection and pip version"; exit 1; }

# Step 4: Sync dependencies
echo "Syncing dependencies with uv..."
uv sync --active || { echo "Failed to sync dependencies. Check pyproject.toml"; exit 1; }

echo "✅ Installation completed successfully!"
