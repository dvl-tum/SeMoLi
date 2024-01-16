#!/bin/bash

# create conda env
# conda env create -f environment.yml
# conda activate SeMoLi

# lapsolver
pip install lapsolver

# pyarrow
pip install pyarrow

# Pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# RAMA
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:$CUDA_HOME/bin
git clone https://github.com/pawelswoboda/RAMA.git
cd RAMA
git submodule update --init --recursive
python setup.py install

# sort for 2D IoU
git clone https://github.com/Jiahao-Ma/2D-3D-IoUs.git
mv 2D-3D-IoUs/cuda_op/ .
cd cuda_op
python setup.py install

# open3d
python3 -m pip install --user open3d

# argoverse 2 api
pip install av2

# chamferdist
pip install chamferdist

# patwork++
conda install cmake
git clone https://github.com/url-kaist/patchwork-plusplus
cp CMakeLists.txt  patchwork-plusplus
cd patchwork-plusplus
sudo apt-get install libeigen3-dev
pip install .

# pyransac
pip install pyransac3d
