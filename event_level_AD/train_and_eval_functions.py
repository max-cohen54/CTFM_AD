import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class EvtDataset(Dataset):
    def __init__(self, evts):  # jets: [N, maxparts, 4]
        self.x = evts[:, :, :3]           # [eta, phi, pt]
        self.mask = evts[:, :, 3:]        # [mask]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx]
    


def cartesian_to_cylindrical(data, eps=1e-8):
    """
    Converts (px, py, pz) → (pt, eta, phi) for a batch of particles.

    Args:
        data (torch.Tensor): [batch_size, 19, 3], where the last dim is (px, py, pz)
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: [batch_size, 19, 3] with (pt, eta, phi)
    """
    px = data[..., 0]
    py = data[..., 1]
    pz = data[..., 2]

    pt = torch.sqrt(px**2 + py**2)
    phi = torch.atan2(py, (px + eps))
    eta = torch.arcsinh(pz / (pt + eps))

    cylindrical = torch.stack([pt, eta, phi], dim=-1)
    return cylindrical


def roc_curve_plot(datasets, plots_path):

    plt.figure(figsize=(10, 8))
    
    bkg_scores = datasets['bkg_test']['AD_scores']

    for tag, data_dict in datasets.items():
        if 'bkg' in tag:
            continue

        sig_scores = data_dict['AD_scores']
        combined_scores = np.concatenate([bkg_scores, sig_scores], axis=0)
        combined_labels = np.concatenate([np.zeros_like(bkg_scores), np.ones_like(sig_scores)], axis=0)

        fpr, tpr, thresholds = roc_curve(combined_labels, combined_scores)
        auroc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{tag} (auc = {auroc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.annotate('better', xy=(0.7, 0.3), xytext=(0.5, 0.1),
                 textcoords='axes fraction', fontsize=12, color='red',
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'{plots_path}/roc_curves.png')
    plt.close()

def plot_features_post_flow(datasets, plots_path):

    raise NotImplementedError("Not implemented")