from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS

#added by Tsach
from scipy.stats import multivariate_normal
from scipy.stats import gamma
import os

#import Batch
from torch_geometric.data import Batch

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


# Code for the CNN used in the paper. It has the same architecture and hyperparameters as the network used
# for spacegroup classification in https://www.nature.com/articles/s41524-023-01164-8, https://github.com/AGI-init/XRDs
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels, in_dim, output_dim):
        super(SimpleConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 80, kernel_size = 100, stride=5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(80, 80, 50, stride=5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(80, 80, 25, stride=2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.flatten = nn.Flatten()
        # Calculate flattened_size dynamically
        self.flattened_size = self._get_flattened_size(input_shape=(1, in_channels, in_dim))
        self.MLP = nn.Sequential(
            nn.Linear(self.flattened_size, 2300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2300, 1150),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1150, output_dim)
        )

    def _get_flattened_size(self, input_shape):
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.shape))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.MLP(x)
        return x

class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        preds = self.encoder(batch)  # shape (N, 1)
        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        loss = F.mse_loss(preds, batch.y)
        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, preds, prefix):
        loss = F.mse_loss(preds, batch.y)
        self.scaler.match_device(preds)
        scaled_preds = self.scaler.inverse_transform(preds)
        scaled_y = self.scaler.inverse_transform(batch.y)
        mae = torch.mean(torch.abs(scaled_preds - scaled_y))

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
        }

        if self.hparams.data.prop == 'scaled_lattice':
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]
            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mae = torch.mean(torch.abs(pred_lengths - batch.lengths))
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mard = mard(batch.angles, pred_angles)

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)
            log_dict.update({
                f'{prefix}_lengths_mae': lengths_mae,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mard': angles_mard,
                f'{prefix}_volumes_mard': volumes_mard,
            })
        return log_dict, loss

class CDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ### Crystalyze Parameters ### 
 
        self.useoriginal = self.hparams.useoriginal # if useoriginal is set, then the original CDVAE generative model is trained. NOTE: This should be False to use Crystalyze 
        self.use_composition_constraint = self.hparams.use_composition_constraint # if true, then the model is trained with composition enforced. NOTE: Crystalyze was NOT trained with this on 
        self.variational_latent_space = getattr(self.hparams, 'variational_latent_space', False) #if true, than a variational latent space is used. NOTE: Crystalyze IS trained with this on

        # Convolutional network parameters
        self.in_dim = getattr(self.hparams, 'in_dim', 8500) # the 1x{input dimension} of the xrd patterns to the CNN. NOTE: Crystalyze is trained with 1x8500 xrd patterns
        self.simple_conv_net = SimpleConvNet(in_channels=1, output_dim=self.hparams.latent_dim, in_dim=self.in_dim) # the convolutional network used to encode the xrd patterns
        
        # Initialize the weights. NOTE: Pre-trained weights are NOT used for the model reported in the paper. 
        if self.in_dim == 8192:
            self.pretrained_weights_path =  "/home/gridsan/tmackey/cdvae/cdvae/pl_modules/model_final_8192.pth" # the path to the pretrained weights for the 8192 xrd patterns
        elif self.in_dim == 8500:
            self.pretrained_weights_path = "/home/gridsan/tmackey/cdvae/cdvae/pl_modules/model_final_8500.pth" # the path to the pretrained weights for the 8500 xrd patterns
        else:
            self.pretrained_weights_path = "" # the path to the pretrained weights for the 256 xrd patterns
        self.use_weight_initialization = False # if true, then the model is trained with pretrained weights
        if self.use_weight_initialization:
            if os.path.exists(self.pretrained_weights_path):
                pretrained_weights = torch.load(self.pretrained_weights_path)
                self.simple_conv_net.load_state_dict(pretrained_weights)
            else:
                print(f"Pretrained weights file not found at {self.pretrained_weights_path}")
        else: 
            print("not using weight initialization")

        # Make sure that the model is trainable 
        for param in self.simple_conv_net.parameters(): 
            param.requires_grad = True

        if torch.cuda.is_available():
            self.simple_conv_net = self.simple_conv_net.to(self.device)

        # Data augmentation parameters
        self.noise_sd = getattr(self.hparams, 'noise_sd', 0.0) # the standard deviation of the noise added to the pseudo-voigt xrd patterns during training. If this is set to -1, the noise is generated
                                                               # from the custom multi-level distributed described in the paper. NOTE: Crystalyze is trained with noise_sd = -1
        print("the noise_sd is: {}".format(self.noise_sd))
        
        ### CDVAE Parameters ###

        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
    
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)
        
        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, MAX_ATOMIC_NUM)

        # for property prediction.
        if self.hparams.predict_property:
            self.fc_property = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                         self.hparams.fc_num_layers, 1)
            
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encode(self, batch, xrd_int, xrd_loc, atom_spec, discrete_simulated_xrd=None, testing=False, pv_xrd=None, multi_hot_encode=None):
        """

        Encode diffraction patterns to latents.

        TODO: remove deprecated arguments.

        Args:
            batch (Tensor): The input batch of data.
            xrd_int (Tensor): The intensity values of the diffraction patterns. Deprecated.
            xrd_loc (Tensor): The location values of the diffraction patterns. Deprecated.
            atom_spec (Tensor): The atomic species information. Deprecated.
            discrete_simulated_xrd (Tensor, optional): The discrete simulated diffraction patterns. Defaults to None. Deprecated.
            testing (bool, optional): Flag indicating if testing mode is enabled. Defaults to False.
            pv_xrd (Tensor, optional): The pseudo Voigt diffraction patterns. Defaults to None.
            multi_hot_encode (Tensor, optional): The multi-hot encoded diffraction patterns. Defaults to None. Deprecated.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The mean, log variance, and latent representation of the input data.
        """

        if self.useoriginal:

            hidden = self.encoder(batch)
            mu = self.fc_mu(hidden)
            log_var = self.fc_var(hidden)
            z = self.reparameterize(mu, log_var)

            return mu, log_var, z

        else:
            if testing:
                print("using psuedo voigt")

            pv_xrd_processed = self.simple_conv_net(pv_xrd)

            if self.variational_latent_space:
                hidden = pv_xrd_processed

                # normalize the hidden vector per row
                hidden = F.normalize(hidden, p=2, dim=1)

                mu = self.fc_mu(hidden)
                log_var = self.fc_var(hidden)
                z = self.reparameterize(mu, log_var)

            else:

                hidden = pv_xrd_processed
                
                z = hidden
                mu = torch.zeros(xrd_loc.size(0), self.hparams.latent_dim, device=self.device)
                log_var = torch.zeros(xrd_loc.size(0), self.hparams.latent_dim, device=self.device)

            return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False, gt_elements = None):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)

        if gt_elements is not None: 
            #impose the composition constraint.
            #The composition constraint only enforces that the elements in the crystal are present in the composition_per_atom tensor
            #It does not enforce the exact number of atoms of each element in the crystal.
            if gt_num_atoms is not None: 
                composition_per_atom = self.composition_constraint(gt_elements, gt_num_atoms, composition_per_atom)
            else:
                composition_per_atom = self.composition_constraint(gt_elements, num_atoms, composition_per_atom)

        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    # This is the same as decode stats, but the number of atoms is sampled from a distribution instead of being predicted
    # NOTE: This is NOT what is used in the Crystalyze model  
    def decode_stats_with_sampling(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                        teacher_forcing=False, gt_elements = None):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            atom_logits = self.fc_num_atoms(z)
            print('sampling from the distribution for the number of atoms')
            num_atoms = torch.multinomial(torch.nn.functional.softmax(atom_logits, dim=-1), 1).squeeze(-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)

        if gt_elements is not None: 
            #impose the composition constraint
            if gt_num_atoms is not None: 
                composition_per_atom = self.composition_constraint(gt_elements, gt_num_atoms, composition_per_atom)
            else:
                composition_per_atom = self.composition_constraint(gt_elements, num_atoms, composition_per_atom)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    def composition_constraint(self, atom_types, num_atoms, composition_per_atom):
            """
            Implicitly restrict the probability distribution from which the atom types are randomly drawn 
            to only include the elements that are present in the crystal.

            atom_types: the atom types in the crystal
            num_atoms: the number of atoms in the crystal
            composition_per_atom: the predicted score per element type for each atom in the crystal. fed into the softmax function

            """

            # Create a range tensor and repeat it as before
            range_tensor = torch.arange(len(num_atoms), device=self.device)
            crystal_ids = torch.repeat_interleave(range_tensor, num_atoms)

            # Convert atom_types into a mask
            atom_mask = atom_types != 0

            # For each unique crystal_id, get its corresponding indices in composition_per_atom
            unique_crystal_ids, counts = torch.unique(crystal_ids, return_counts=True)

            composition_per_atom = composition_per_atom + 1

            start_idx = 0
            for u_id, count in zip(unique_crystal_ids, counts):
                relevant_elements = atom_types[u_id][atom_mask[u_id]]

                #first step: create a hugely negative additive mask 
                mask = torch.ones_like(composition_per_atom[start_idx])
                mask *= (-10**6) # creating a matrix like [-10^6, ..., -10^6]
                mask[relevant_elements-1] = 0 # setting the elements that are present in the crystal to 0

                # second step: create a second additive mask that is used to boost any small scores for the correct elements
                additive_mask_for_normalization = torch.zeros_like(composition_per_atom[start_idx]) # creating a matrix like [0, ..., 0]
                additive_mask_for_normalization[relevant_elements-1] = 0.0001 # setting the elements that are present in the crystal to 0.0001

                # Apply masks to the relevant segment of composition_per_atom
                composition_per_atom[start_idx:start_idx+count] += mask # adding the huge negative mask to the relevant segment of composition_per_atom
                composition_per_atom[start_idx:start_idx+count] += additive_mask_for_normalization # adding the small additive mask to the relevant segment of composition_per_atom

                # Update start index for next iteration
                start_idx += count
            return composition_per_atom

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None, gt_elements_input=None, sampling = False):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
 
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        if not self.use_composition_constraint: 
            gt_elements_input = None

        if sampling: 
             #use the sampling version of decode stats. NOTE: This is NOT what is used in the Crystalyze model
            num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats_with_sampling(
                z, gt_num_atoms, gt_elements=gt_elements_input)
        else: 
            num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
                z, gt_num_atoms, gt_elements=gt_elements_input)
        
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        composition_per_atom = F.softmax(composition_per_atom, dim=-1)

        if gt_atom_types is None:
            composition_per_atom = composition_per_atom.cuda(0)
            num_atoms = num_atoms.cuda(0)
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles, gt_elements=gt_elements_input)
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                    'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                    'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def generate_background_noise(self, pv_xrd):
        """
        Generates background noise for pseudo-Voigt XRD patterns by sampling noise in a 2-stage process. 
        The noise is sampled from a gamma distribution, the parameters of which are determined by a multivariate normal distribution with 
        predefined mean and covariance based on the noise characteristics of the training data. 

        Steps:
        
        1. **Multivariate Normal Sampling**:
        - Samples are drawn from a multivariate normal distribution with predefined mean 
            and covariance based on training data noise characteristics.
        
        2. **Filtering**:
        - Samples are filtered to remove values where:
            - The second column exceeds `ln(0.05)`.
            - The first column exceeds `ln(10)`.
        - The filtered samples are truncated to match the size of `pv_xrd`.

        3. **Exponentiation and Gamma Sampling**:
        - The filtered samples are exponentiated.
        - Gamma distributions are generated using these values, and samples are drawn.
        
        Parameters:
        ----------
        pv_xrd : torch.Tensor
            The pseudo-Voigt XRD pattern tensor for noise addition.

        Returns:
        -------
        gamma_samples : torch.Tensor
            A tensor of noise samples, shaped to match the input XRD pattern.
        """
            
        # the mean and covariate of the noise was found by fitting a multivariate normal distribution to the noise in the training data
        mean = np.array([1.06709083, -5.39670499]) 
        cov = np.array([[0.19984825, 0.14809546],
                        [0.14809546, 1.31019437]])

        # Create an instance of a multivariate normal distribution
        mvn = multivariate_normal(mean, cov)

        # Sample points from the distribution
        num_samples = 500  # You can change this to your desired number of samples
        samples = mvn.rvs(size=num_samples)

        #remove any samples with values in the second column greater than ln(0.05)
        #remove any samples with values in the first column greater than ln(10)
        samples_filtered = samples[samples[:,1] < np.log(0.05)]
        samples_filtered = samples_filtered[samples_filtered[:,0] < np.log(10)]

        #take the first 256 
        samples_filtered = samples_filtered[:pv_xrd.shape[0]]

        #calculate the exponentials of the samples
        samples_exp = np.exp(samples_filtered)

        a = samples_exp[:, 0]
        scale = samples_exp[:, 1]
        #make a gamma distribution for each a and scale
        gamma_samples = []
        for i in range(len(a)):
            gamma_samples.append(gamma(a = a[i], scale = scale[i], loc = 0).rvs(8500))
        
        gamma_samples = np.stack(gamma_samples)
        gamma_samples = torch.tensor(gamma_samples, device=self.device)

        #unsqueeze at the 1st dimension
        gamma_samples = gamma_samples.unsqueeze(1)

        return gamma_samples

    def forward(self, batch, teacher_forcing, training):

        batch_reserve = batch
        xrd_int = batch_reserve[1]
        xrd_loc = batch_reserve[2]
        atom_spec = batch_reserve[3]
        disc_sim_xrd = batch_reserve[4]
        pv_xrd = batch_reserve[5]
        multi_hot_encoding = batch_reserve[6]

        # if a noise standard deviation is provided, add noise to the pseudo-voigt xrd patterns
        if self.noise_sd > 0.0:
            pv_xrd = pv_xrd + torch.randn_like(pv_xrd, device=self.device) * self.noise_sd
        
        # if the noise standard deviation is set to -1, generate background noise for the pseudo-voigt xrd patterns
        # using the 2-stage process described in the generate_background_noise function
        elif self.noise_sd == -1:
            gamma_samples = self.generate_background_noise(pv_xrd)
            pv_xrd[:, :8500] = pv_xrd[:, :8500] + gamma_samples

        batch = batch[0]
        
        mu, log_var, z = self.encode(batch, xrd_int, xrd_loc, atom_spec,
                                      discrete_simulated_xrd = disc_sim_xrd, 
                                      pv_xrd = pv_xrd, 
                                      testing = False, 
                                      multi_hot_encode = multi_hot_encoding)
        
        kld_loss = self.kld_loss(mu, log_var)

        if self.use_composition_constraint: 
            gt_elements = atom_spec
        else: 
            gt_elements = None

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
        pred_composition_per_atom) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing, gt_elements)
        
        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                        (batch.num_atoms.size(0),),
                                        device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))

        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])

        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms, 
            pred_lengths, pred_angles, gt_elements, dropout = 0, is_training = True)
        
        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
            
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                used_type_sigmas_per_atom, batch)
        
        if self.hparams.predict_property:
            property_loss = self.property_loss(z, batch)
        else:
            property_loss = 0.

        diffraction_loss = 0 #legacy from old experiments

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'diffraction_loss': diffraction_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z
        }
    
    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        if loss < 100000000: 
            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return loss
        else:
            print("the loss is too high, skipping this batch") #this was implemented in the case of low-quality training data, it is not relevant for the application of Crystalyze as implemented 
            return None
 
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        try: 
            outputs = self(batch, teacher_forcing=False, training=False)
            log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
            self.log_dict(
                log_dict,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return loss
        except Exception as e: 
            print(f"An error occurred during validation at batch {batch_idx}: {e}")
            return None

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        try: 
            outputs = self(batch, teacher_forcing=False, training=False)
            log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
            self.log_dict(
                log_dict,
            )
            return loss
        except Exception as e:
            print(f"An error occurred during testing at batch {batch_idx}: {e}")
            return None
        
    def compute_stats(self, batch, outputs, prefix):
        batch_reserve = batch
        batch = batch[0]

        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']
        diffraction_loss = outputs['diffraction_loss']  # legacy from old experiments

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss + 
            diffraction_loss) # legacy from old experiments

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_diffraction_loss': diffraction_loss,  # legacy from old experiments
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss)

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()
