import sys 
sys.path.append('../src') 
from pathlib import Path
import numpy as np
import KPCA
from utils import get_data, log_exp



def save_modes_cavity(Re, Wi, beta, kernel_type, n_modes = 10, eps=None, case ='cavity_ref', dx = 0.0125, stepsize = None):
    theta = (1.0-beta)/(Re*Wi)
    if case == 'cavity_ref':
        X, _ = get_data(Re, Wi, beta, case='cavity_ref', n_data=-2, dir_path='../../npz_data/dataset_cavity')
    else:
        X, _ = get_data(Re, Wi, beta, eps = eps, case='fene-p', n_data=-2, dir_path='../../npz_data/dataset_fenep')

    if stepsize is not None:
        X = X[...,stepsize-1::stepsize]
    # Create KPCA reduction
    kernel = KPCA.KernelPCA()
    kernel.fit(X.T, n_components=n_modes, kernel=kernel_type, theta = theta,degree=1, eps=eps, dx=dx)
    Phi = kernel.transform(X.T, theta,eps=eps, dx = dx)

    # Save modes
    sim_type = 'oldroyd' if case == 'cavity_ref' else 'fenep'
    filename = f'saved_modes/cavity_modes_{sim_type}_Re_{Re:g}_Wi_{Wi:g}_beta_{beta:g}_kernel_{kernel_type}_nmodes_{n_modes}_'
    filename_KPCA = f'saved_modes/cavity_KernelPCA_{sim_type}_Re_{Re:g}_Wi_{Wi:g}_beta_{beta:g}_kernel_{kernel_type}_nmodes_{n_modes}_'
    if eps is not None:
        filename += f'eps_{eps:g}_'
        filename_KPCA += f'eps_{eps:g}_'
    np.savez_compressed(filename + '.npz', Phi=Phi, allow_pickle=True)
    kernel.save_model(filename_KPCA + '.npz', compressed = True)
    #Log the action
    MANIFEST = Path("summary/manifest.csv")
    exp_id = f'modes_cavity_oldroyd_Re_{Re:g}_Wi_{Wi:g}_beta_{beta:g}_kernel_{kernel_type}_nmodes_{n_modes}'
    rec = log_exp.RunRecord(
        exp_id=exp_id,
        geom="cavity",
        constitutive_model="Oldroyd-B",
        kernel=kernel_type,
        task="create_modes",
        Wi=Wi,
        beta=beta,
        r=n_modes,
        notes="Create data for reduction"
    )
    log_exp.upsert_manifest(rec, MANIFEST)

def load_modes(Re, Wi, beta, kernel_type, n_modes = 10, eps = None, sim_type = 'oldroyd'):
    filename = f'saved_modes/cavity_modes_{sim_type}_Re_{Re:g}_Wi_{Wi:g}_beta_{beta:g}_kernel_{kernel_type}_nmodes_{n_modes}_'
    if eps is not None:
        filename += f'eps_{eps:g}_'
    Phi = np.load(filename+'.npz', allow_pickle=True)['Phi']
    return Phi