# 2.5DAttention
This repository is the implementation of our paper "Interpretable 2D Vision Models for 3D Medical Images". 

## Setup
 - Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
 - Execute ```conda env create -f environment.yml```
 - ```conda activate twoandahalfdimensions```

## Running the Code
 - Put the data into a data folder
    - MedMNIST data is downloaded automatically when code is executed
    - Other classification data can be structured following the [Torch ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) way. We note that depending on the data format the code must be extended to support the respective file types.
  - Define the config file for your use case
    - You find example configs in the config folder
    - These are structured as a base config (for each method that we support) + an extension config for the respective dataset
    - An explanation for each config argument can be found in the [config readme](configs/README.md).
  - Run the training via ```python twoandahalfdimensions/train.py -cn <config_name.yaml>```
    - The code uses [hydra](https://hydra.cc/) for argument handling, thus all config attributes can be modified or added on the command line. 