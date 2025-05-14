import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from sklearn.model_selection import train_test_split
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 14
import os


filemap = {
    "bkg":"/eos/home-m/mmcohen/chop_or_not_development/data/background_for_training.h5",
    "a4l":"/eos/home-m/mmcohen/chop_or_not_development/data/Ato4l_lepFilter_13TeV.h5",
    "htaunu":"/eos/home-m/mmcohen/chop_or_not_development/data/hChToTauNu_13TeV_PU20.h5",
    "htautau":"/eos/home-m/mmcohen/chop_or_not_development/data/hToTauTau_13TeV_PU20.h5",
    "lq":"/eos/home-m/mmcohen/chop_or_not_development/data/leptoquark_LOWMASS_lepFilter_13TeV.h5",
}

def transform_and_normalize(data, mean_std=None, eps=1e-8, pxpypz=True):
    """
    Transforms (pt, eta, phi) to (px, py, pz) and normalizes per particle over the batch.
    Only unmasked particles (id != 0) are used to compute mean/std.
    
    Args:
        data (torch.Tensor): [batch_size, 19, 4] where last dim is (pt, eta, phi, id)
        mean_std (tuple, optional): Tuple of (mean, std), each of shape [19, 3]
        eps (float): Small value to avoid division by zero
        
    Returns:
        norm_data (torch.Tensor): [batch_size, 19, 3], normalized px, py, pz
        mean (torch.Tensor): [19, 3], per-particle mean
        std (torch.Tensor): [19, 3], per-particle std
    """
    pt = data[..., 0]
    eta = data[..., 1]
    phi = data[..., 2]
    pid = data[..., 3]  # ID feature

    # Mask for valid (unmasked) particles: shape [batch_size, 19]
    mask = pid != 0

    # Compute Cartesian coordinates
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    cartesian = np.stack([px, py, pz], axis=-1)  # [batch_size, 19, 3]

    if mean_std is None:
        # Expand mask to match cartesian shape
        expanded_mask = mask[..., np.newaxis]  # [batch_size, 19, 1]
        masked_cartesian = cartesian * expanded_mask

        # Count valid entries per particle
        valid_counts = expanded_mask.sum(axis=0, keepdims=False).clip(min=1)  # [19, 1]

        mean = masked_cartesian.sum(axis=0) / valid_counts  # [19, 3]
        var = ((masked_cartesian - mean)**2 * expanded_mask).sum(axis=0) / valid_counts
        std = np.sqrt(var + eps)  # [19, 3]
    else:
        mean, std = mean_std

    norm_data = (cartesian - mean) / std
    # Reattach pid as 4th channel
    norm_data = np.concatenate([norm_data, pid[..., np.newaxis]], axis=-1)  # [batch_size, 19, 4]

    return norm_data, mean, std

def load_and_preprocess(p_train=0.5, p_test=0.25, plots_path=None, pxpypz=True):
    datasets = {tag: {} for tag in filemap.keys()}

    print('Extracting...')

    for tag in filemap.keys():
        f = h5py.File(filemap[tag], 'r')
        datasets[tag]['data'] = f["Particles"][:].astype(np.float32)[:,:,:]

        datasets[tag]['data'][:,:,2] = datasets[tag]['data'][:,:,2] - datasets[tag]['data'][:,0,2,np.newaxis]
        datasets[tag]['data'][:,:,2] = datasets[tag]['data'][:,:,2] + 2.*np.pi*(datasets[tag]['data'][:,:,2]<-np.pi).astype(np.float32) - 2.*np.pi*(datasets[tag]['data'][:,:,2]>np.pi).astype(np.float32)
        print('Loaded',tag)

    # Plot some events from each dataset
    if plots_path is not None:
        iev = 0
        fig,axs = plt.subplots(2,5,figsize=(24,12),sharex=True,sharey=True)

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
    idxs = np.arange(len(datasets['bkg']['data']))
    train_idxs, _idxs = train_test_split(idxs, train_size=p_train, random_state=42)
    test_size = p_test / (1 - p_train)
    test_idxs, val_idxs = train_test_split(_idxs, train_size=test_size, random_state=42)
    datasets['bkg_train'] = {key: value[train_idxs] for key, value in datasets['bkg'].items()}
    datasets['bkg_test'] = {key: value[test_idxs] for key, value in datasets['bkg'].items()}
    datasets['bkg_val'] = {key: value[val_idxs] for key, value in datasets['bkg'].items()}
    del datasets['bkg']

    # Normalize the data
    if pxpypz:
        for tag in datasets.keys():
            # First, scale and compute mean/std over train set
            scaled_train_data, mean, std = transform_and_normalize(datasets[tag]['data'])
            datasets['bkg_train']['data'] = scaled_train_data

            # Then, apply same scaling with same mean/std to test/val sets
            scaled_test_data, _, _ = transform_and_normalize(datasets['bkg_test']['data'], mean_std=(mean, std))
            datasets['bkg_test']['data'] = scaled_test_data
            scaled_val_data, _, _ = transform_and_normalize(datasets['bkg_val']['data'], mean_std=(mean, std))
            datasets['bkg_val']['data'] = scaled_val_data

    else:
        mean_pt = np.mean(datasets['bkg_train']['data'][:, :, 0], axis=0, keepdims=True)
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
        plt.hist(datasets['bkg_train']['data'][:,:,0].flatten()[datasets['bkg_train']['data'][:,:,-1].flatten()>0.5], bins=50, histtype='step', density=True, label='feature 1')
        plt.hist(datasets['bkg_train']['data'][:,:,1].flatten()[datasets['bkg_train']['data'][:,:,-1].flatten()>0.5], bins=50, histtype='step', density=True, label='feature 2')
        plt.hist(datasets['bkg_train']['data'][:,:,2].flatten()[datasets['bkg_train']['data'][:,:,-1].flatten()>0.5], bins=50, histtype='step', density=True, label='feature 3')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{plots_path}/feature_histograms.png')
        plt.close()

    return datasets




        

            
