#!/bin/bash
# =============================================================================
# Environment setup script for EvSSC (or VoxFormer) with mmdetection3d
# This script creates a conda environment, installs PyTorch 1.9.1 (CUDA 11.1),
# mmcv-full 1.4.0, mmdetection3d v0.17.1, and compiles deform_attn_3d ops.
# =============================================================================

# 1. Create conda environment with Python 3.8
conda create -n open-mmlab python=3.8 -y

# 2. Activate the environment (works with different shell configurations)
eval "$(conda shell.bash hook)"
conda activate open-mmlab

# 3. Install PyTorch 1.9.1 with CUDA 11.1 support
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 4. Install gcc-6 (required for compiling custom CUDA operators)
conda install -c omgarcia gcc-6 -y

# 5. Install mmcv-full 1.4.0 (compatible with PyTorch 1.9 and CUDA 11.1)
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 6. Install mmdetection and mmsegmentation at specific versions
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# 7. Clone and install mmdetection3d from source (v0.17.1)
if [ ! -d "mmdetection3d" ]; then
    git clone https://github.com/open-mmlab/mmdetection3d.git
fi
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .   # -v: verbose, -e: editable mode
cd ..

# 8. Install timm (PyTorch image models)
pip install timm

# 9. Clone the target repository (EvSSC in this example)
git clone https://github.com/Pandapan01/EvSSC.git
cd EvSSC

# 10. Build deformable 3D attention CUDA ops
cd deform_attn_3d
python setup.py build_ext --inplace
cd ../..

echo "============================================================"
echo "Installation completed successfully!"
echo "Activate environment with: conda activate open-mmlab"
echo "Navigate to EvSSC: cd EvSSC"
echo "============================================================"