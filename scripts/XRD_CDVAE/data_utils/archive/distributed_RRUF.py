import importlib
import data_utils.RRUFF_utils
importlib.reload(data_utils.RRUFF_utils)
from data_utils.RRUFF_utils import *
import data_utils.pv_utils
import sys

def main():
    worker_num = int(sys.argv[1])
    num_splits = int(sys.argv[2])
    global_path = os.path.join("/home/gridsan/tmackey/cdvae/scripts/XRD_CDVAE/RRUFF_Testing", "ckpt/amcsd/2024-03-20/unfiltered_xy_cif_dict.pt")
    unfiltered_compounds = torch.load(global_path)

    start_index = int(len(unfiltered_compounds) * worker_num / num_splits)
    end_index = int(len(unfiltered_compounds) * (worker_num + 1) / num_splits)

    print("worker_num", worker_num, "start_index", start_index, "end_index", end_index)

    subset = list(unfiltered_compounds.items())[start_index:end_index]

    # Initialize the X-ray diffraction (XRD) calculator
    xrd_calculator = XRDCalculator(wavelength='CuKa')  # Using Copper K-alpha radiation by default

    filtered_compounds = []
    for index in tqdm(range(len(subset))):
        try: 
            structure = subset[index][1][2]

            # Calculate the diffraction pattern
            pattern = xrd_calculator.get_pattern(structure)

            #extract the experimental pattern 
            x = subset[index][1][0][:,0][:-1]
            y = subset[index][1][0][:,1][:-1]
            y = y/np.max(y)

            xy_merge = np.column_stack((pattern.x, pattern.y))

            U, V, W = 0.1, 0.1, 0.1 
            sim_xrd = data_utils.pv_utils.simulate_pv_xrd_for_row(xy_merge, U, V, W, noise=0.0)

            cossim = data_utils.pv_utils.cosine_similarity(sim_xrd, y)

            print(cossim)
            
            if cossim > 0.75: 
                new_value = (subset[index], cossim)
                filtered_compounds.append(new_value)
                
            #viz.plot_xrd(pattern, x, y)
        except: 
            continue

    filename = "filtered_by_sim_" + str(worker_num) + ".pt"
    filename = os.path.join("/home/gridsan/tmackey/cdvae/scripts/XRD_CDVAE/RRUFF_Testing/comp_results", filename)
    torch.save(filtered_compounds, filename)
    
if __name__ == '__main__':
    main()