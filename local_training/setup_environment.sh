#!/bin/bash
# Setup script for Megatron-LM local training on RTX 5080
# This script creates a conda environment and installs all required dependencies

set -e

echo "=== Megatron-LM Local Training Setup ==="
echo ""

# Configuration
ENV_NAME="megatron-lm"
PYTHON_VERSION="3.11"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "Detected GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "Installing PyTorch with CUDA support..."
# Install PyTorch 2.4+ with CUDA 12.4 support (compatible with RTX 5080)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "Installing Megatron-LM dependencies..."
pip install \
    ninja \
    packaging \
    transformers \
    datasets \
    sentencepiece \
    tiktoken \
    wandb \
    tensorboard \
    pybind11 \
    regex \
    einops \
    flask-restful \
    nltk \
    pytest

echo ""
echo "Installing Megatron-LM..."
cd "$(dirname "$0")/.."
pip install -e .

echo ""
echo "Building Megatron-LM helpers (for dataset processing)..."
cd megatron/core/datasets
make || python -c "from megatron.core.datasets.utils import compile_helpers; compile_helpers()"
cd -

echo ""
echo "Installing NVIDIA Apex (optional but recommended for better performance)..."
pip install packaging
git clone https://github.com/NVIDIA/apex /tmp/apex 2>/dev/null || true
cd /tmp/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    . || echo "Warning: Apex installation failed. Training will still work without it."
cd -

echo ""
echo "Installing Flash Attention 2 (highly recommended for RTX 5080)..."
pip install flash-attn --no-build-isolation || echo "Warning: Flash Attention installation failed."

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Download and preprocess data: ./local_training/prepare_data.sh"
echo "  2. Start training: ./local_training/train.sh"
