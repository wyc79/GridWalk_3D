#!/bin/bash

# qrsh -l h_data=4G,gpu,RTX2080Ti


# create folders to store output in scratch
mkdir -p $SCRATCH/deeplabcut/GridWalk_3D/main_view_result/AD
mkdir -p $SCRATCH/deeplabcut/GridWalk_3D/main_view_result/WT

# modules_lookup -m 
module load python/3.9.6
module load cuda/11.0
module load cudnn/8.1.1

# general packages, upgrades
python3 -m pip install --upgrade pip

# go to scratch folder to install deeplabcut
cd $SCRATCH/deeplabcut
git clone -l -s https://github.com/DeepLabCut/DeepLabCut.git cloned-DLC-repo
cd cloned-DLC-repo/
python3 -m pip install ".[tf]"

export PATH=$PATH:/u/home/x/xwei/.local/bin