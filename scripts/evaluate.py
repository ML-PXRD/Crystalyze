import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model
from torch.nn import functional as F
import os
import numpy as np

def new_dataloader_batch_processor(batch): 
    """
    Process a batch of data from a dataloader.
    """

    batch_reserve = batch
    xrd_int = batch_reserve[1]
    xrd_loc = batch_reserve[2]
    atom_spec = batch_reserve[3]
    batch = batch[0]
    disc_sim_xrd = batch_reserve[4]
    pv_xrd = batch_reserve[5]
    multi_hot_encode = batch_reserve[6]

    return batch_reserve, xrd_int, xrd_loc, atom_spec, batch, disc_sim_xrd, pv_xrd, multi_hot_encode

def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1, num_batches=15):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    predicted_property = []
    input_data_list = []

    if num_batches < len(loader):
        #randomly sample num_batches indices from 0 to len(loader)
        rng = np.random.default_rng(seed=42) # want to have the batch indices be the same between evals
        batch_indices_to_use = rng.choice(len(loader), num_batches, replace=False)
        print("using batch indices: ", batch_indices_to_use)
    else:
        batch_indices_to_use = range(len(loader))

    for idx, batch in enumerate(loader):
        if idx in batch_indices_to_use:
            batch_reserve, xrd_int, xrd_loc, atom_spec, batch, disc_sim_xrd, pv_xrd, multi_hot_encoding = new_dataloader_batch_processor(batch)

            #put everything on the gpu
            if torch.cuda.is_available():
                xrd_int = xrd_int.cuda()
                xrd_loc = xrd_loc.cuda()
                atom_spec = atom_spec.cuda()
                disc_sim_xrd = disc_sim_xrd.cuda()
                batch = batch.cuda()
                pv_xrd = pv_xrd.cuda()
                multi_hot_encoding = multi_hot_encoding.cuda()    
                
            print(f'batch {idx} in {len(loader)}')
            batch_all_frac_coords = []
            batch_all_atom_types = []
            batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
            batch_lengths, batch_angles = [], []
            batch_predicted_property = []

            for eval_idx in range(num_evals):
                # NOTE: None is fed into the legacy parameter of "batch" for the crystal graph 
                _, _, z = model.encode(None, xrd_int, 
                                       xrd_loc, 
                                       atom_spec, 
                                       disc_sim_xrd, 
                                       testing = True, 
                                       pv_xrd = pv_xrd, 
                                       multi_hot_encode = multi_hot_encoding)

                #predict the property 
                # NOTE: This is try-excepted as a patch for a feature that was not completed and has yet to be 
                # fully removed from the codebase. Properties prediction is not the focus of this work.
                try: 
                    property = model.fc_property(z)
                    print("property: ", property)
                except:
                    property = torch.tensor([0.0])
                
                gt_num_atoms = batch.num_atoms if force_num_atoms else None
                gt_atom_types = batch.atom_types if force_atom_types else None
                if gt_num_atoms is not None:
                    print("using gt_num_atoms")
                
                if gt_atom_types is not None:
                    print("using gt_atom_types")
                    
                outputs = model.langevin_dynamics(
                    z, ld_kwargs, gt_num_atoms, gt_atom_types, atom_spec)

                # collect sampled crystals in this batch.
                batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
                batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
                batch_atom_types.append(outputs['atom_types'].detach().cpu())
                batch_lengths.append(outputs['lengths'].detach().cpu())
                batch_angles.append(outputs['angles'].detach().cpu())
                batch_predicted_property.append(property.detach().cpu())

                if ld_kwargs.save_traj:
                    batch_all_frac_coords.append(
                        outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                    batch_all_atom_types.append(
                        outputs['all_atom_types'][::down_sample_traj_step].detach().cpu())
                    
            # collect sampled crystals for this z.
            frac_coords.append(torch.stack(batch_frac_coords, dim=0))
            num_atoms.append(torch.stack(batch_num_atoms, dim=0))
            atom_types.append(torch.stack(batch_atom_types, dim=0))
            lengths.append(torch.stack(batch_lengths, dim=0))
            angles.append(torch.stack(batch_angles, dim=0))
            predicted_property.append(torch.stack(batch_predicted_property, dim=0))

            if ld_kwargs.save_traj:
                all_frac_coords_stack.append(
                    torch.stack(batch_all_frac_coords, dim=0))
                all_atom_types_stack.append(
                    torch.stack(batch_all_atom_types, dim=0))

            # Save the ground truth structure
            input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    predicted_property = torch.cat(predicted_property, dim=1)

    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, predicted_property, input_data_batch)

# NOTE: generation and optimization are not the focus of this work.
def generation(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)

# NOTE: generation and optimization are not the focus of this work.
def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        batch_reserve, xrd_int, xrd_loc, atom_spec, batch = new_dataloader_batch_processor(batch)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys-1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=('recon' in args.tasks) or
        ('opt' in args.tasks and args.start_from == 'data'), test_set_override=args.test_set_override)
    print(test_loader)
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack, predicted_property, input_data_batch) = reconstructon(
            test_loader, model, ld_kwargs, args.num_evals,
            args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step, num_batches = args.num_batches)

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        #check to see if model_path / recon_out_name exists
        #if it does, then we need to increment the name from eval_recon_{args.label}.pt to eval_recon_{args.label}_1.pt
        #and so on
        #TODO: this makes overwriting old results difficult, should change to have a iteration flag or something
        if os.path.exists(model_path / recon_out_name):
            i = 1
            while os.path.exists(model_path / recon_out_name):
                recon_out_name = f'eval_recon_{args.label}_{i}.pt'
                i += 1
        
        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'predicted_property': predicted_property,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'opt' in args.tasks:
        print('Evaluate model on the property optimization task.')
        start_time = time.time()
        if args.start_from == 'data':
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(model, ld_kwargs, loader)
        optimized_crystals.update({'eval_setting': args,
                                   'time': time.time() - start_time})

        if args.label == '':
            gen_out_name = 'eval_opt.pt'
        else:
            gen_out_name = f'eval_opt_{args.label}.pt'
        torch.save(optimized_crystals, model_path / gen_out_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    #if you want to force atom types you would input --force_atom_types
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    #number of batches to evaluate
    parser.add_argument('--num_batches', default=36, type=int)
    parser.add_argument('--test_set_override', default=None, type=str)
    
    args = parser.parse_args()

    main(args)
