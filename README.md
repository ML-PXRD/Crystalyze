# Crystalyze Repository

This repository contains the code and resources for the Crystalyze project.

## Overview

Crystalyze is a machine learning model that combines XRD representation learning with a crystal diffusion variational autoencoder (CDVAE) to generate crytal structures given an x-ray diffraction pattern. 

## Installation

To use Crystalyze, follow these steps:

1. conda env create -f env_sub.yml
2. conda activate crystalyze
3. pip install -e .

## Usage

To train and use the Crystalyze, refer to the following instructions:

To train the model, run the following command:

```python cdvae/run.py data=$1 expname=$2 max_num_atoms=$3```

where: 
- `$1` is the path to the data directory
- `$2` is the name that you want to give to the run 
- `$3` is the maximum number of atoms in the crystal structures


To generate crystal structures, run 

```python scripts/evaluate.py --model_path $1 --tasks recon --num_batches $2 $FORCE_NUM_ATOMS_FLAG $FORCE_ATOM_TYPES_FLAG --label $6 ```

where:
- `$1` is the path to the model
- `$2` is the number of batches
- `$6` is the label for the run
- `$FORCE_NUM_ATOMS_FLAG` is a flag to force the number of atoms in the generated crystal structures to be the same as the number of atoms in the dummy graph data input
- `$FORCE_ATOM_TYPES_FLAG` is a flag to force the atom types in the generated crystal structures to be the same as the atom types in the dummy graph data input

Example:

``` python scripts/evaluate.py --model_path /home/gridsan/tmackey/hydra/singlerun/2024-01-31/ae_pf/ --tasks recon --num_batches 1 --force_num_atoms --force_atom_types --test_set_override "unsolved_compounds" --label "wstoich_unsolved_compounds" ```


You can find a demo on how to do this for a single crystal and xrd pattern in the `notebooks` directory.