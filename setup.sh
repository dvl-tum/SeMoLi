#!/bin/bash

# create conda env
#conda env create -f environment.yml
#conda activate SeMoLi

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

# av2 update
cp PseudoDetection3D/av2_update/cuboid.py $conda_path/envs/SeMoLi/lib/python3.8/site-packages/av2/structures/
cp PseudoDetection3D/av2_update/av2_sensor_dataloader.py $conda_path/envs/SeMoLi/lib/python3.8/site-packages/av2/datasets/sensor/
cp PseudoDetection3D/evaluation/av2_evaluation_update/detection/* $conda_path/envs/SeMoLi/lib/python3.8/site-packages/av2/evaluation/detection/
