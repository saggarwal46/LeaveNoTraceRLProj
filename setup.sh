#!/bin/bash

#install Coach
cd coach
git submodule init
git submodule update

cd ..
cp -f install.sh requirements_coach.txt ./coach/
cd coach

./install.sh

# Install Other dependencies
pip3 install gym==0.9.3
pip3 install mujoco-py==0.5.7

# Copy mujoco files
cd ./../mujoco_files
mkdir -p ~/.mujoco
cp -r * ~/.mujoco/

# sudo apt-get install nvidia-384
# sudo apt-get install cuda-8-0

pip install --ignore-installed box2d-py
pip install --ignore-installed pachi-py

pip3 install tensorflow-gpu==1.14
pip install tensorboardX
pip install scipy==1.1.0 pandas==0.20.2 Pillow==8.4.0 pygame==1.9.3
# pip3 install tensorflow==1.4.1 --ignore-installed
# pip3 install tensorflow-gpuy==1.15 
# pip install numpy==1.13.0
pip install numpy==1.14.5


# conda update --all -y
conda install -c conda-forge  glew glfw mesalib=23.0.0 -y

conda install cudatoolkit=10.0
conda install cudnn=7.3.1
#for installing cuda_8.0.61_375.26_linux-run    
# export $PERL5LIB