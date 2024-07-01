import sys 
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
import numpy as np
from KPCA import *
import glob
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
    dspath = '/home/fabio/npz_data/KPCA_4roll'
    files = glob.glob('4_roll6*', root_dir=dspath)
    X = []
    P = []
    n_data = 3000 * len(files)

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
    n_components = 20
    matrixes = [np.memmap(f'{dspath}/X_{x}.npz',dtype='float32', mode='w+', shape=(n_data,n_components)) for x in kpca_files]
    

    filename = path.join(dspath, 'Center_Mat.dat')

    Center = np.memmap(filename, dtype='float32', mode='r', shape=(n_data,n_data))
    print('Create Center Matrix Completed')
    print()
    for f_m, M, kernel in zip(mat_files, matrixes,kpca_files):

        print(f'Starting {kernel}...', flush=True)
        K = np.memmap(f_m,dtype='float32', mode='r',shape=(n_data,n_data))

        K_row = np.mean(K, axis=0)
        K_col = np.mean(K, axis=1)[:,None]
        K_all = np.mean(K)
        K_centered = K - K_row - K_col + K_all
        K[:] = K_centered
        # K_centered = K - np.mean(K, axis=1).reshape(-1, 1)
        
        print(f'Centered Kernel Completed')
        # Eigen decomposition
        print(f'Starting Eigendecomp...', flush=True)
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        print(f'Finish Eigendecomp')


        # Sort eigenvalues and eigenvectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Select top n_components eigenvectors
        eigenvectors = eigenvectors[:, :n_components]
        eigenvectors_normalized = eigenvectors / np.sqrt(eigenvalues[:n_components])
        print(f'Starting KPCA multiplication...', flush=True)
        X_kpca = np.dot(K, eigenvectors_normalized)
        print(f'Finish KPCA multiplication')

        M[:] = X_kpca
        M.flush()

        print(f'Starting U_fit multiplication...', flush=True)
        U_fit = Center@eigenvectors_normalized
        print(f'Finish U_fit multiplication')

        np.savez(f'{dspath}/U_fit_{kernel}.npz', U =U_fit, eigenvalues = eigenvalues)
        print(f'Finish {kernel}', flush=True)
        print()