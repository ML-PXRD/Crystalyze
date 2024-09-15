import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)

from typing import Literal, List, Optional, Tuple, Dict 
import numpy as np

class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, 
                 train_fraction: ValueNode = 1,
                 max_num_atoms: ValueNode = 20,
                 source: ValueNode = "any",
                 num_augmented_data: ValueNode = 0,
                 **kwargs):
        
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.train_fraction = train_fraction
        self.max_num_atoms = max_num_atoms
        self.source = source
    
        self.num_augmented_data = num_augmented_data

        if self.num_augmented_data == 0:
            self.cached_data = preprocess(self.path,
                                                    train_fraction=self.train_fraction,
                                                    prop_list=[prop])
            add_scaled_lattice_prop(self.cached_data, lattice_scale_method)

        else:
            self.cached_data = []
            for i in range(self.num_augmented_data):
                print('processing augmented data', i)
                self.cached_data.append(preprocess(self.path,
                                                    train_fraction=self.train_fraction,
                                                    prop_list=[prop],
                                                    index = i))
                add_scaled_lattice_prop(self.cached_data[i], lattice_scale_method) #this technically only needs to be done once but the crystal data
                                                                                   #is the same for each augmented dataset so it doesn't matter
        
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        if self.num_augmented_data:
            return len(self.cached_data[0])
        else:
            return len(self.cached_data)

    def __getitem__(self, index):
        if self.num_augmented_data == 0:
            data_dict = self.cached_data[index]
        else:
            #randomly choose one of indexes from the list
            dataset_to_choose = self.cached_data[np.random.randint(0, self.num_augmented_data)]
            data_dict = dataset_to_choose[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        xrd_intensities = data_dict['xrd_intensities']
        xrd_locations = data_dict['xrd_locations']
        atomic_species = data_dict['atomic_species']
        disc_sim_xrd = data_dict['disc_sim_xrd']
        multi_hot_encoding = data_dict['multi_hot_encoding']
        pv_xrd = data_dict['pv_xrd']

        # if there is only 1 pv xrd, unsqueeze it
        if len(pv_xrd.shape) == 1:
            pv_xrd = pv_xrd.unsqueeze(0)

        #0 out the first 1000 columns of pv_xrd. This makes the actual 2theta range 15-90
        pv_xrd[:, :1000] = 0

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        return data, xrd_intensities, xrd_locations, atomic_species, disc_sim_xrd, pv_xrd, multi_hot_encoding

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"

class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )

    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
