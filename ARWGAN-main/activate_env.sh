#!/bin/bash
# ARWGAN 虛擬環境啟動腳本
export PATH=~/miniconda/bin:$PATH
source ~/miniconda/bin/activate arwgan
cd /mnt/nvme/p3/Project/arwgan/ARWGAN-Project/ARWGAN-main
echo "虛擬環境已啟動！"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
