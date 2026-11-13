#!/bin/bash
set -x
W=/tmp/yzhang176
PY=$W/envs/ad/bin/python
PIP="$PY -m pip"
MIR=https://pypi.tuna.tsinghua.edu.cn/simple
cd $W
curl -sL --max-time 60 -o $W/get-pip.py https://bootstrap.pypa.io/get-pip.py || curl -sL -o $W/get-pip.py $MIR/../../..
$PY $W/get-pip.py -i $MIR --no-cache-dir
$PIP install -U pip -i $MIR --no-cache-dir
$PIP install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision
$PIP install --no-cache-dir -i $MIR "opencv-python==4.11.0.86" matplotlib Pillow numpy scipy scikit-learn faiss-gpu-cu12 tqdm tifffile PyYAML pandas "timm>=1.0.20" pytest
echo "=== VERIFY ==="
$PY -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
$PY -c "import faiss,cv2,skimage;print('faiss',faiss.__version__ if hasattr(faiss,'__version__') else 'ok','cv2',cv2.__version__)"
echo "=== SETUP_DONE ==="
df -h / | tail -1