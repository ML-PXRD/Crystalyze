# Training Data for Crystalyze

After downloading the data from the shared drive, you will want to move it into this directory.
The data is from the [MP-20](mp_20) (Jain et al., 2013) dataset, specifically from the mp_20 data used in the [CDVAE](cdvae) (Xie et al., 2021) paper.

The crystal graph data structures are computed with code from the cdvae moduleThe process is identical to the one used in the CDVAE paper but with the graph conversion done once ahead of time to cut down on overhead per run. The crystal graph data is identical between the augmented and non-augmnented datasets.

The xrd data is calculated with pymatgen. The preprocessing notebook/code will be released soon. The process for the creation of both the augmented and non-augmented datasets is as described in the paper. 

## Citations

CDVAE 

```
@article{xie2021crystal,
  title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
  author={Xie, Tian and Fu, Xiang and Ganea, Octavian-Eugen and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2110.06197},
  year={2021}
}
```

MP_20:

```
@article{jain2013commentary,
  title={Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
  author={Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and others},
  journal={APL materials},
  volume={1},
  number={1},
  pages={011002},
  year={2013},
  publisher={American Institute of PhysicsAIP}
}
```

