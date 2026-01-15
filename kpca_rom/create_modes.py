import sys 
sys.path.append('../src') 
from pathlib import Path
import numpy as np
import KPCA
from utils import get_data, log_exp



def save_modes_cavity(Re, Wi, beta, kernel_type, n_modes = 10):
    theta = (1.0-beta)/(Re*Wi)
    X, _ = get_data(Re, Wi, beta, case='cavity_ref', n_data=-2, dir_path='../../npz_data/dataset_cavity')


    # Create KPCA reduction
    kernel = KPCA.KernelPCA()
    kernel.fit(X.T, n_components=n_modes, kernel=kernel_type, theta = theta,degree=1)
    Phi = kernel.transform(X.T, theta)

    # Save modes
    filename = f'saved_modes/cavity_modes_oldroyd_Re_{Re:g}_Wi_{Wi:g}_beta_{beta:g}_kernel_{kernel_type}_nmodes_{n_modes}_.npz'
    np.savez_compressed(filename, Phi=Phi, allow_pickle=True)

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

def load_modes(Re, Wi, beta, kernel_type, n_modes = 10):
    filename = f'saved_modes/cavity_modes_oldroyd_Re_{Re:g}_Wi_{Wi:g}_beta_{beta:g}_kernel_{kernel_type}_nmodes_{n_modes}_.npz'
    Phi = np.load(filename, allow_pickle=True)['Phi']
    return Phi