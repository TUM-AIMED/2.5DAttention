# 2.5DAttention
This repository is the implementation of our paper [Interpretable 2D Vision Models for 3D Medical Images](https://arxiv.org/abs/2307.06614) and was used for [Private, fair and accurate: Training large-scale, privacy-preserving AI models in medical imaging](https://arxiv.org/abs/2302.01622). 

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

## Citation

### In case you use this repository, please consider citing the corresponding papers:


Ziller, A., Erdur, A. C., Trigui, M., Güvenir, A., Mueller, T. T., Müller, P., Jungmann, F., ... & Kaissis, G. (2023). Explainable 2D Vision Models for 3D Medical Data. arXiv preprint arXiv:2307.06614. https://doi.org/10.48550/arXiv.2307.06614

    @misc{ziller2023interpretable,
        title={Interpretable 2D Vision Models for 3D Medical Images}, 
        author={Alexander Ziller and Ayhan Can Erdur and Marwa Trigui and Alp Güvenir and Tamara T. Mueller and Philip Müller and Friederike Jungmann and Johannes Brandt and Jan Peeken and Rickmer Braren and Daniel Rueckert and Georgios Kaissis},
        year={2023},
        eprint={2307.06614},
    }


S. Tayebi Arasteh, A. Ziller, C. Kuhl, M. Makowski, S. Nebelung, R. Braren, D. Rueckert, D. Truhn, G. Kaissis. "*Private, fair and accurate: Training large-scale, privacy-preserving AI models in medical imaging*". ArXiv, arxiv.2302.01622, https://doi.org/10.48550/arxiv.2302.01622, 2023.

    @article {DPCXR2023,
      author = {Tayebi Arasteh, Soroosh and Ziller, Alexander and Kuhl, Christiane and Makowski, Marcus and Nebelung, Sven and Braren, Rickmer and Rueckert, Daniel and Truhn, Daniel and Kaissis, Georgios},
      title = {Private, fair and accurate: Training large-scale, privacy-preserving AI models in medical imaging},
      year = {2023},
      doi = {10.48550/ARXIV.2302.01622},
      publisher = {arXiv},
      URL = {https://arxiv.org/abs/2302.01622},
      journal = {arXiv}
    }