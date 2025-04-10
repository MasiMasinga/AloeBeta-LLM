#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Install NVIDIA drivers and CUDA
sudo apt-get install -y nvidia-driver-525
sudo apt-get install -y nvidia-cuda-toolkit

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install huggingface-cli
pip install huggingface_hub

# Verify GPU is accessible
nvidia-smi
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 