#!/bin/bash

# create conda env
# conda env create -f environment.yml
# conda activate SeMoLi

# lapsolver
python3 -m pip install lapsolver

# pyarrow
python3 -m pip install pyarrow

# Pytorch3d
python3 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"

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
python3 -m pip install av2

# chamferdist
python3 -m pip install chamferdist

# patwork++
conda install cmake
git clone https://github.com/url-kaist/patchwork-plusplus
cp CMakeLists.txt  patchwork-plusplus
cd patchwork-plusplus
sudo apt-get install libeigen3-dev
python3 -m pip install .

# pyransac
python3 -m pip install pyransac3d

# install waymo open dataset
python3 -m pip install waymo-open-dataset-tf-2-11-0==1.6.1