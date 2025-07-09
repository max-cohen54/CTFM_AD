import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from sklearn.model_selection import train_test_split
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 14
import os


def load_subdicts_from_h5(save_dir, tags_to_use=None):
    """
    Loads sub-dictionaries of NumPy arrays from HDF5 files in a directory and reconstructs the original structure.
    
    Args:
        save_dir (str): The directory where the HDF5 files are stored.
    
    Returns:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
    """
    main_dict = {}
    
    for filename in os.listdir(save_dir):
        if filename.endswith(".h5") and not filename.startswith('.'):

            
            sub_dict_name = os.path.splitext(filename)[0]
            if tags_to_use is not None and sub_dict_name not in tags_to_use:
                continue
            
            file_path = os.path.join(save_dir, filename)
            with h5py.File(file_path, 'r') as f:
                sub_dict = {key: np.array(f[key]) for key in f}
            main_dict[sub_dict_name] = sub_dict
            print(f"Loaded {sub_dict_name} from {file_path}")
    
    return main_dict



def transform_and_normalize(data, mean_std=None, eps=1e-8, pxpypz=True):
    """
    Transforms (pt, eta, phi) to (px, py, pz) and normalizes per particle over the batch.
    Only unmasked particles (id != 0) are used to compute mean/std.
    
    Args:
        data (torch.Tensor): [batch_size, 15, 4] where last dim is (pt, eta, phi, id)
        mean_std (tuple, optional): Tuple of (mean, std), each of shape [15, 3]
        eps (float): Small value to avoid division by zero
        
    Returns:
        norm_data (torch.Tensor): [batch_size, 15, 3], normalized px, py, pz
        mean (torch.Tensor): [15, 3], per-particle mean
        std (torch.Tensor): [15, 3], per-particle std
    """
    pt = data[..., 0]
    eta = data[..., 1]
    phi = data[..., 2]
    pid = data[..., 3]  # ID feature

    # Mask for valid (unmasked) particles: shape [batch_size, 15]
    mask = pid != 0

    # Compute Cartesian coordinates
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    cartesian = np.stack([px, py, pz], axis=-1)  # [batch_size, 15, 3]

    if mean_std is None:
        # Expand mask to match cartesian shape
        expanded_mask = mask[..., np.newaxis]  # [batch_size, 15, 1]
        masked_cartesian = cartesian * expanded_mask

        # Count valid entries per particle
        valid_counts = expanded_mask.sum(axis=0, keepdims=False).clip(min=1)  # [15, 1]

        mean = masked_cartesian.sum(axis=0) / valid_counts  # [15, 3]
        var = ((masked_cartesian - mean)**2 * expanded_mask).sum(axis=0) / valid_counts
        std = np.sqrt(var + eps)  # [15, 3]
    else:
        mean, std = mean_std

    norm_data = (cartesian - mean) / std
    # Reattach pid as 4th channel
    norm_data = np.concatenate([norm_data, pid[..., np.newaxis]], axis=-1)  # [batch_size, 15, 4]

    return norm_data, mean, std

def load_and_preprocess(p_train=0.5, p_test=0.25, plots_path=None, pxpypz=True):
    datasets = load_subdicts_from_h5('/pscratch/sd/m/mcohen54/data/unpreprocessed_L1', tags_to_use=None)

    # Add a particle type feature to the data
    for tag, data_dict in datasets.items():
        # Get original data shape
        N = data_dict['data'].shape[0]
        
        # Create particle ID mask array
        pid_mask = np.zeros((N, 15))
        
        # Set masks for each particle type
        # Jets (first 6 positions)
        pid_mask[:, 0:6] = np.where(data_dict['data'][:, 0:6, 0] != 0, 1, 0)
        
        # Taus (next 4 positions) 
        pid_mask[:, 6:10] = np.where(data_dict['data'][:, 6:10, 0] != 0, 2, 0)
        
        # Muons (next 4 positions)
        pid_mask[:, 10:14] = np.where(data_dict['data'][:, 10:14, 0] != 0, 3, 0)
        
        # MET (last position)
        pid_mask[:, 14] = np.where(data_dict['data'][:, 14, 0] != 0, 4, 0)
        
        # Add pid_mask as 4th feature
        data_dict['data'] = np.concatenate([data_dict['data'], 
                                          pid_mask[..., np.newaxis]], 
                                          axis=-1)

    # Plot some events from each dataset
    if plots_path is not None:
        iev = 0
        fig,axs = plt.subplots(2,8,figsize=(24,12),sharex=True,sharey=True)

        for j in range(2):  # Loop over two events
            for i, (tag, data_dict) in enumerate(datasets.items()):  # Loop over each tag
                data = data_dict['data']
                axs[j, i].scatter(
                    data[iev+j, :, 1][data[iev+j, :, -1] > 0.],  # x-axis: eta
                    data[iev+j, :, 2][data[iev+j, :, -1] > 0.],  # y-axis: phi
                    s=data[iev+j, :, 0][data[iev+j, :, -1] > 0.] * 10.,  # size: scaled by the first feature
                    alpha=0.5,
                    color=f'C{i}'
                )
                if j == 0:
                    axs[j, i].set_title(tag)
                if j == 1:
                    axs[j, i].set_xlabel(r'$\eta$')
                if i == 0:
                    axs[j, i].set_ylabel(r'$\phi$')
                axs[j, i].set_xlim(-5., 5.)
                axs[j, i].set_ylim(-np.pi, np.pi)
        plt.tight_layout()
        plt.savefig(f'{plots_path}/event_maps.png')
        plt.close(fig)

    # split data in train/test/val
    idxs = np.arange(len(datasets['EB']['data']))
    train_idxs, _idxs = train_test_split(idxs, train_size=p_train, random_state=42)
    test_size = p_test / (1 - p_train)
    test_idxs, val_idxs = train_test_split(_idxs, train_size=test_size, random_state=42)
    datasets['EB_train'] = {key: value[train_idxs] for key, value in datasets['EB'].items()}
    datasets['EB_test'] = {key: value[test_idxs] for key, value in datasets['EB'].items()}
    datasets['EB_val'] = {key: value[val_idxs] for key, value in datasets['EB'].items()}
    del datasets['EB']

    # Normalize the data
    if pxpypz:
        # First, scale and compute mean/std over train set
        scaled_train_data, mean, std = transform_and_normalize(datasets['EB_train']['data'])
        datasets['EB_train']['data'] = scaled_train_data
        
        # Then, apply same scaling with same mean/std to all the other datasets
        for tag in datasets.keys():
            if tag == 'EB_train': continue

            scaled_data, _, _ = transform_and_normalize(datasets[tag]['data'], mean_std=(mean, std))
            datasets[tag]['data'] = scaled_data

    else:
        mean_pt = np.mean(datasets['EB_train']['data'][:, :, 0], axis=0, keepdims=True)
        std0 = np.std(data_dict['data'][:, :, 0].flatten()[data_dict['data'][:, :, -1].flatten()>0.5])
        std1 = np.std(data_dict['data'][:, :, 1].flatten()[data_dict['data'][:, :, -1].flatten()>0.5])
        std2 = np.std(data_dict['data'][:, :, 2].flatten()[data_dict['data'][:, :, -1].flatten()>0.5])


        for tag, data_dict in datasets.items():
            # transform to log(pt) to help symmetrize
            data_dict['data'][:,:,0] = np.nan_to_num(np.log(data_dict['data'][:,:,0]), neginf=0.)

            data_dict['data'][:,:,0] = data_dict['data'][:,:,0] - mean_pt
            data_dict['data'] = data_dict['data']/np.array([[[std0,std1,std2,1.]]])

    for tag, data_dict in datasets.items():
        print(f'{tag}: {data_dict["data"].shape}')

    
    # Plot histogram of features
    if plots_path is not None:
        plt.figure(figsize=(12, 8))
        plt.hist(datasets['EB_train']['data'][:,:,0].flatten()[datasets['EB_train']['data'][:,:,-1].flatten()>0.5], bins=50, histtype='step', density=True, label='feature 1')
        plt.hist(datasets['EB_train']['data'][:,:,1].flatten()[datasets['EB_train']['data'][:,:,-1].flatten()>0.5], bins=50, histtype='step', density=True, label='feature 2')
        plt.hist(datasets['EB_train']['data'][:,:,2].flatten()[datasets['EB_train']['data'][:,:,-1].flatten()>0.5], bins=50, histtype='step', density=True, label='feature 3')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{plots_path}/feature_histograms.png')
        plt.close()

    return datasets