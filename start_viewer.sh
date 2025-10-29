#!/bin/bash

# YCB Viewer Startup Script
# This script sets up the environment and starts the YCB viewer

echo "YCB Object Viewer - Starting up..."

# Check if conda environment exists
if ! conda env list | grep -q "sdr-dashboard"; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
fi

# Activate the environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sdr-dashboard

# Check if void scene exists, generate if not
if [ ! -f "assets/scenes/examiner/void_black.glb" ]; then
    echo "Generating void scene for rendering..."
    mkdir -p assets/scenes/examiner
    python scripts/generate_void_scene.py
fi

# Check if YCB assets are available
if [ ! -d "assets/ycb/meshes" ]; then
    echo "YCB meshes not found. You may need to download them."
    echo "Try running: python scripts/download_ycb.py"
    echo "Continuing anyway - some objects may not load properly."
fi

# if grep -qi "microsoft" /proc/version 2>/dev/null; then
#     echo "WSL environment detected, forcing CPU renderer"
#     export YCB_VIEWER_FORCE_CPU=1
#     # Force Habitat/Magnum to use OSMesa software rendering when GPU is disabled
#     export MAGNUM_GPU_DRIVER=none
# fi

echo "Starting YCB Object Viewer..."
python ycb_viewer.py

echo "YCB Viewer has exited."