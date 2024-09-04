# AIDA workshop on advanced data loading techniques

This repository contains code for the AIDA Technical Workshop on advanced data loading using NVIDIA's [DALI](https://developer.nvidia.com/dali) framework and pre-augmented dataset with numpy memory mapped arrays.

The code is in the `/example` directory and has two main code bases. The `train_pet_resnet*` files are all variations of a simple image classification script where we look at how we can use DALI to offload data augmentation to the GPU, as well as pre-augment images to a static file.

The second set of examples is `unet_training_array.py` and its companion `unet_create_dataset.py`. These are intended for the second part of the workshop where we try our hands at making a complex pipeline use DALI.

## Preparations

There are two ways to run the code of this workshop. The best experience is from running the code on your local computer by first cloning this repository and then install the dependencies using conda/mamba. The other is to use the suplied jupyter notebook as a kind of terminal and run the different scripts on Colab. This has the advantage that you don't have to have a computer with a mamba installation and a decent GPU.

### Local installation

**N.b. This option assumes that you have a local NVIDIA GPU installed, for other options you need to consult the pytorch documentation.**

Start by cloning this repository:

```shell
$ git clone https://github.com/eryl/aida-dali-workshop.git
```

If you don't have conda or mamba installed, we suggest you install it using [Miniforge](https://github.com/conda-forge/miniforge). This is a minimal conda distribution with mamba preinstalled. mamba is a drop-in replacement for conda, which has _much_ better dependency resolution, making package installations far less annoying. Be aware that it uses the conda-forge conda channel instead of the official one. This means the packages are more bleeding edge, for good and bad. They are also not curated, and can be a potential security risk (similar to installing packages from PyPI).


After you have a conda installation, create and activate the environment for this workshop by running:

```shell
$ mamba env create -f environment.yml
$ mamba activate aidali
```

or if you don't have mamba installed:

```shell
$ conda env create -f environment.yml
$ conda activate aidali
```

#### Running the scripts

All the scripts used in this workshop are in the separate directories. The main scripts are `train.py` and `analyze_predictions.py`. The scripts typically assume that they are run from the repository root directory, e.g. like `python step_1/train.py`.


### Colab installation
To run these on Google Colab, use the following notebook as an entry point and follow the instructions there: [Colab notebook](https://colab.research.google.com/github/eryl/aida-dali-workshop/blob/main/run_on_colab.ipynb).


## The initial script

The first script we'll look at is in `examples/train_pet_resnet.py`. It's a simple training script which uses parallel PyTorch data loader to load data. Run the script with

```shell
$ python examples/train_pet_resnet.py
```
