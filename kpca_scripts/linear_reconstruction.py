import sys 
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
import numpy as np
from KPCA import *
import glob
import argparse
import os.path as path

DX = 1/(2**6)
PCA_KWD = {'kernel_type':'linear', 'theta':None, 'eps':None, 'dx' : DX, 'dy' : DX}
OLD_KWD = {'kernel_type':'oldroyd', 'theta':None, 'eps':None, 'dx' : DX, 'dy' : DX}
POL_KWD = {'kernel_type':'poly', 'theta':None, 'eps':3, 'dx' : DX, 'dy' : DX}
RBF_KWD = {'kernel_type':'rbf', 'theta':None, 'eps':1, 'dx' : DX, 'dy' : DX}
COS_KWD = {'kernel_type':'cosine', 'theta':None, 'eps':None, 'dx' : DX, 'dy' : DX}

def get_matrix(filename):
    #reads the file
    f_split = filename.split('_')
    Re = float(f_split[2].replace('Re',''))
    Wi = float(f_split[3].replace('Wi',''))
    beta = float(f_split[4].replace('beta',''))
    fields = np.load(f'{dspath}/{filename}', allow_pickle=True)["fields"].item()
    param =  np.repeat((Re,Wi,beta), 3000).reshape((3,-1)).T
    #Extract the fields
    u = fields["vel-u"]
    v = fields["vel-v"]
    Bxx = fields["Bxx"]
    Bxy = fields["Bxy"]
    Byy = fields["Byy"]
    q = np.stack((u,v,Bxx, Bxy, Byy), axis=-1)


    # reshape for the expected code format
    TU = q[:,:,:,0].reshape((q.shape[0]**2, q.shape[2]))
    TV = q[:,:,:,1].reshape((q.shape[0]**2, q.shape[2]))
    T11 = q[:,:,:,2].reshape((q.shape[0]**2, q.shape[2]))
    T12 = q[:,:,:,3].reshape((q.shape[0]**2, q.shape[2]))
    T22 = q[:,:,:,4].reshape((q.shape[0]**2, q.shape[2]))
    T = np.concatenate((TU, TV, T11,T12,T22), axis=1).reshape(-1, q.shape[2]) # by column axis=1(intercal..), by row axis=0
    theta_sqrt = np.sqrt((1-param[:,2])/(param[:,0] * param[:,1])).reshape((-1,1))
    X =  T[:,:3000]
    return X.T, theta_sqrt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Latent', '-d', default=20, type=int, help="Latent dimension") 

    args = parser.parse_args()
    dspath = '/home/fabio/npz_data/KPCA_4roll'
    files = glob.glob('4_roll6*', root_dir=dspath)
    X = []
    P = []

    mat_files = [   
        f'{dspath}/Kernel_oldroyd.npz',
        f'{dspath}/Kernel_linear.npz',
        f'{dspath}/Kernel_poly.npz',
        f'{dspath}/Kernel_rbf.npz',
        f'{dspath}/Kernel_cosine.npz'
        ]
    kpca_files = [   
        f'oldroyd',
        f'linear',
        f'poly',
        f'rbf',
        f'cosine'
        ]
    k_args = [
        OLD_KWD,
        PCA_KWD,
        POL_KWD,
        RBF_KWD,
        COS_KWD
    ]
    total_components = 20
    n_components = args.Latent
    nfiles= 12
    n_data = 3000 * nfiles
    npoints = 4096*5
    degree = 1

    PCA_MAT = [np.memmap(f'{dspath}/X_{x}.npz',dtype='float32', mode='r', shape=(n_data,total_components))[:,:n_components] for x in kpca_files]
    datset_matrix = np.memmap(f'{dspath}/dataset.dat',dtype='float32', mode='r', shape=(n_data, npoints))
    datset_theta = np.memmap(f'{dspath}/dataset_theta.dat',dtype='float32', mode='r', shape=(n_data,1))    


    ones_fit = np.ones((n_data,1), dtype='float32')


    print(datset_theta.shape, datset_theta.min(), np.count_nonzero(datset_theta))
    print(datset_matrix[:2])
    print(datset_theta[:2])
    K_centered = np.memmap(f'{dspath}/Kernel_Center_temp.dat',dtype='float32', mode='w+',shape=(n_data,n_data))
    for M, kernel in zip(PCA_MAT,kpca_files):

        print(f'Starting {kernel}...', flush=True)
        Q = np.concatenate([ones_fit]+ [ones_fit / datset_theta ]  + [(M)**(k+1) for k in range(degree)] + [(M / datset_theta)**(k+1) for k in range(degree)], axis=1)
        R, _, _, _ = np.linalg.lstsq(Q, datset_matrix, rcond=None)

        print(f'Ending {kernel}...', flush=True)
        print(f'Saving {kernel}...', flush=True)
        R_save = np.memmap(f'{dspath}/R_{kernel}_{n_components}_Modes.dat',dtype='float32', mode='w+', shape=R.shape)
        reconst = np.memmap(f'{dspath}/reconst_{kernel}_{n_components}_Modes.dat',dtype='float32', mode='w+', shape=(n_data, npoints))
        R_save[:] = R[:]
        R_save.flush()

        reconst[:] = Q @ R
        reconst.flush()
        print(f'Done Saving {kernel}...', flush=True)
        