#!/usr/bin/env bash
# BenDFM PointNeXt Setup
# Builds CUDA extensions for PointNeXt (optional but recommended)
# Requires: conda environment already activated

set -e

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: No conda environment active. Run: conda activate bendfm"
    exit 1
fi

echo "=== Building PointNeXt CUDA Extensions ==="
echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Update git submodules
echo "Updating git submodules..."
git submodule update --init --recursive
git submodule update --remote --merge
echo "✓ Submodules updated"
echo ""

# Detect GPU compute capability
echo "Detecting GPU..."
GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_ARCH=$(echo $GPU_COMPUTE | tr -d '.')

echo "GPU Compute Capability: $GPU_COMPUTE"
echo "GPU Architecture: sm_$GPU_ARCH"
echo ""
echo "Building CUDA extensions... (may take 2-5 minutes)"
echo ""

# Build PointNeXt CUDA extension
cd benchmark/models/PointNeXt/openpoints/cpp/pointnet2_batch
export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
CUDA_HOME=/usr/local/cuda-12.6 \
python setup.py install
cd - > /dev/null

echo ""
echo "✓ CUDA extensions installed successfully"
echo "Ready to train: python benchmark/train/pointnext.py --help"
