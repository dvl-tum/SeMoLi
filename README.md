# 3DOpenWorldMOT

## Installation
Create a conda environment from environment file
```
conda env create -f environment.yml
```

```
pip install lapsolver
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

```
export conda_path=~/anaconda3/
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:$CUDA_HOME/bin
git clone https://github.com/pawelswoboda/RAMA.git
cd RAMA
git submodule update --init --recursive
python setup.py install
```

```
git clone https://github.com/Jiahao-Ma/2D-3D-IoUs.git
mv 2D-3D-IoUs/cuda_op/ 3DOpenWorldMOT/
cd 3DOpenWorldMOT/cuda_op
python setup.py install
```

```
python3 -m pip install --user open3d
```
