# 3DOpenWorldMOT

## Installation
Create a conda environment from environment file
```
bash setup_conda.sh
conda env create -f environment.yml
```

Install additional packages using
```
bash setup.sh
```

Update av2 api by running in your conda environment where ```<conda_path>``` is the place where your anaconda is installed, e.g., ```~/anaconda3/```
```
export conda_path=<conda_path>
cp PseudoDetection3D/av2_update/cuboid.py $conda_path/envs/SeMoLi/lib/python3.8/site-packages/av2/structures/
cp PseudoDetection3D/av2_update/av2_sensor_dataloader.py $conda_path/envs/SeMoLi/lib/python3.8/site-packages/av2/datasets/sensor/
cp PseudoDetection3D/evaluation/av2_evaluation_update/detection/* $conda_path/envs/SeMoLi/lib/python3.8/site-packages/av2/evaluation/detection/
```


## Data Preparation
The downloaded data will be stored to this directories base directory in the data directory. To set the required environment variable run:
```
export BASE_DIR=$(pwd)
```

To download and convert Waymo Open Dataset run the follwing:
```
cd Waymo_Preparation
bash download_and_extract_all.sh
cd ..
```
This will download the whole tfrecord files, but will only extract the LiDAR data.

To download Argoverse2 dataset run the following:
```
conda install s5cmd -c conda-forge
export BASE_DIR="<base_dir>"
bash download_av2.sh
```
This will only download the lidar data, 3D annotations, and ego vehicle poses of the AV2 dataset. Since camera data is not needed in this project we aviod downloading it.

## Trajectory Estimation
For the trajectory estimation replace ```<split>``` by either train or val and ```<dataset>``` by either waymo or av2 and run the following:
```
cd NTF_lidar/
bash tools/preprocess_pcs_<split>_<dataset>.sh
bash tools/compute_flow_<split>_<dataset>.sh
cd ../
```
Since the trajectory estimation is a rather long process, you can also strart several processes at the same time if you are using a slurm-based system or something similar by specifying the ```from_``` and ```to_``` flags in the preprocssing which refers to the sequence count. Similarly, for the flow computation ```from_``` and ```to_``` flags can be specified.

## Pseudo-Label Model Training
