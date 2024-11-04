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

def get_matrix(filename, ndata = 3000):
    #reads the file
    filename_no_ext = "".join(filename.split('.')[:-1])
    f_split = filename_no_ext.split('_')
    Re = float(f_split[2].replace('Re',''))
    Wi = float(f_split[3].replace('Wi',''))
    beta = float(f_split[4].replace('beta',''))
    fields = np.load(f'{dspath}/{filename}', allow_pickle=True)["fields"].item()
    param =  np.repeat((Re,Wi,beta), ntimes).reshape((3,-1)).T
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
    X =  T[:,1:ndata+1]
    return X.T, theta_sqrt

if __name__ == '__main__':
    # dspath = '/home/fabio/npz_data/KPCA_4roll' # Four roll
    dspath = '/home/fabio/npz_data/dataset_cavity' # cavity

    kernel_path = '/home/fabio/npz_data/Kernels'
    os.makedirs(kernel_path, exist_ok=True)
    files = glob.glob('*.npz', root_dir=dspath)
    X = []
    P = []
    ntimes = 100
    n_data = ntimes * len(files)

    mat_files = [   
        f'{kernel_path}/Kernel_oldroyd.npz',
        f'{kernel_path}/Kernel_linear.npz',
        f'{kernel_path}/Kernel_poly.npz',
        f'{kernel_path}/Kernel_rbf.npz',
        f'{kernel_path}/Kernel_cosine.npz'
        ]
    

    Nx = Ny = 80
    npoints = (Nx * Ny)*5
    nfiles = len(files)
    datset_matrix = np.memmap(f'{dspath}/dataset.dat',dtype='float32', mode='w+', shape=(ntimes*nfiles, npoints))
    datset_theta = np.memmap(f'{dspath}/dataset_theta.dat',dtype='float32', mode='w+', shape=(ntimes*nfiles,1))
    for i in range(len(files)):
        X1, t = get_matrix(files[i], ntimes)
        datset_matrix[i*ntimes:(i+1)*ntimes,:] = X1[:]
        datset_theta[i*ntimes:(i+1)*ntimes,:] = t[:]
        datset_matrix.flush()
        datset_theta.flush()



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
    matrixes = [np.memmap(x,dtype='float32', mode='w+', shape=(n_data,n_data)) for x in mat_files]

    for i in range(len(files)):
        X1, t1 = get_matrix(files[i], ntimes)
        for j in range(i,len(files)):
            X2, t2 = get_matrix(files[j], ntimes)

            for k_opt,M in zip(k_args, matrixes):
                print(i,j,k_opt['kernel_type'])
                if k_opt['kernel_type'] == 'oldroyd':
                    k_opt['theta'] = t1 @ t2.T
                K = compute_kernel_matrix(X1, X2,**k_opt)
                M[i*ntimes:(i+1)*ntimes, j*ntimes:(j+1)*ntimes] = K
                if i != j:
                    M[j*ntimes:(j+1)*ntimes, i*ntimes:(i+1)*ntimes] = K.T

                M.flush()

    

    # filename = path.join(dspath, 'Center_Mat.dat')
    # Center = np.memmap(filename, dtype='float32', mode='r', shape=(n_data,n_data))
    # print('Create Center Matrix Completed')
    # print()
    K_centered = np.memmap(f'{kernel_path}/Kernel_Center_temp.npz',dtype='float32', mode='w+',shape=(n_data,n_data))
    for f_m, M, kernel in zip(mat_files, matrixes,kpca_files):

        print(f'Starting {kernel}...', flush=True)
        K = np.memmap(f_m,dtype='float32', mode='r',shape=(n_data,n_data))

        K_row = np.mean(K, axis=0)
        K_col = np.mean(K, axis=1)[:,None]
        K_all = np.mean(K)
        K_centered[:] = K - K_row - K_col + K_all
        K_centered.flush()
        # K_centered = K - np.mean(K, axis=1).reshape(-1, 1)    
        
        print(f'Centered Kernel Completed')
        # Eigen decomposition
        print(f'Starting Eigendecomp...', flush=True)
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        print(f'Finish Eigendecomp')


        # Sort eigenvalues and eigenvectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Select top n_components eigenvectors
        eigenvectors = eigenvectors[:, :n_components]
        eigenvectors_normalized = eigenvectors / np.sqrt(eigenvalues[:n_components])
        print(f'Starting KPCA multiplication...', flush=True)
        X_kpca = eigenvectors * np.sqrt(eigenvalues[:n_components])
        print(f'Finish KPCA multiplication')

        M[:] = X_kpca
        M.flush()

        print(f'Starting U_fit multiplication...', flush=True)
        U_fit = eigenvectors_normalized
        print(f'Finish U_fit multiplication')

        mean = K_row[None,:]
        np.savez(f'{kernel_path}/U_fit_{kernel}.npz', U =U_fit, eigenvalues = eigenvalues, eigenvectors = eigenvectors, mean = mean)
        print(f'Finish {kernel}', flush=True)
        print()