NOTE: We are still in the process of migrating the raw data for Crystalyze to this repository and fixing dependency issues. We will update this README as soon as the data is migrated in the near future!

# Crystalyze Repository

This repository contains the code and resources for the Crystalyze project. This code is built on top of the excellent [Crystal Diffusion Variational Autoencoder](https://github.com/txie-93/cdvae) repository and uses the XRD CNN model architecture from the [UnifiedML XRD repository](https://github.com/AGI-init/XRDs/tree/main)

## Overview

Crystalyze is a machine learning model that combines XRD representation learning with a crystal diffusion variational autoencoder (CDVAE) to generate crytal structures given an x-ray diffraction pattern. 

## Installation

To use Crystalyze, follow these steps:

1. Install the cdvae conda environment by running `conda env create -f env.yml`. 
    * Note: cpu support should work across all major systems and gpu support has been confirmed on NVIDIA CUDA 12.2. NVIDIA CUDA 12.4 does not seem to work with the current environment's pytorch version.
    We are working on updating the environment to work with CUDA 12.4, also feel free to post or reach out if you find a solution. 

2. conda activate cdvae

3. pip install -e .

4. Copy the .env.template file, rename it to .env, and fill in the necessary information.

## Data 

Result data discussed in the paper as well as the data used to train the model can be found in this google drive folder: [link](https://drive.google.com/drive/u/0/folders/1iANYLKp4pscNSA1VirSSSrPnt-2BNfzx) NOTE: the link is not yet active, but will be made public once the repository is fully online. 

The data is organized as follows:

- `Unsolved_compounds`: contains the xrd and cif information for the compounds from the PDF and Freedman lab that were solved using Crystalyze

- `mp_20`: contains the xrd and structure information for the materials project dataset (extended from the `mp_20` dataset in the original CDVAE repository to pre-calculate the graph data and xrd patterns for this specific task)

- `mp_20_augmented`: contains the augmented xrd data for the materials project dataset 

- `model_folder`: contains the checkpoint and hyperparameters for the model trained on the augmented materials project dataset

Download the data. Move mp_20 and mp_20_augmented into the data directory in Crystalyze (this step is only required if you want to train / test on this data). 

## Usage

To train and use the Crystalyze, refer to the following instructions:

To train the model, run the following command:

```python cdvae/run.py data=$1 expname=$2 max_num_atoms=$3```

where: 
- `$1` is the name of the directory in data that your training data is in 
- `$2` is the name that you want to give to the run 
- `$3` is the maximum number of atoms in the crystal structures (set to 20 in general, functionality to adjust this to other values is on the way)

Ex: 

```python cdvae/run.py data=mp_20_final expname=myexp max_num_atoms=20```

To generate crystal structures, run 

```python scripts/evaluate.py --model_path $1 --tasks recon --num_batches $2 $FORCE_NUM_ATOMS_FLAG $FORCE_ATOM_TYPES_FLAG --label $3 ```

where:
- `$1` is the path to the model
- `$2` is the number of batches
- `$3` is the label for the run
- `$FORCE_NUM_ATOMS_FLAG` is a flag to force the number of atoms in the generated crystal structures to be the same as the number of atoms in the dummy graph data input
- `$FORCE_ATOM_TYPES_FLAG` is a flag to force the atom types in the generated crystal structures to be the same as the atom types in the dummy graph data input

## Future Releases 
* Code for computing metrics given generated crystals
* Integrated diffusion + StructSnap inference notebook tutorial. 
