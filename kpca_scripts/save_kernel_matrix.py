import sys 
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
import numpy as np
from KPCA import *
import glob

if __name__ == '__main__':
    dspath = '../../npz_data/KPCA_4roll'
    files = glob.glob('*.npz', root_dir=dspath)
    X = []
    P = []
    for f in files:
        #reads the file
        f_split = f.split('_')
        Re = float(f_split[2].replace('Re',''))
        Wi = float(f_split[3].replace('Wi',''))
        beta = float(f_split[4].replace('beta',''))
        fields = np.load(f'{dspath}/{f}', allow_pickle=True)["fields"].item()
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

        X.append(T[:,:3000])
        P.append(param)

    X = np.concatenate(X, axis = 1).T
    P = np.vstack(P)


    # COMPUTE KERNEL MATRIX
    dx = dy = 1/(2**6)
    theta_sqrt = np.sqrt((1-P[:,2])/(P[:,0] * P[:,1])).reshape((-1,1))
    theta_matrix = theta_sqrt @ theta_sqrt.T
    
    #PCA
    eps = None
    K = compute_kernel_matrix(X, X,'linear', None, eps,dx,dy)
    np.savez(f'{dspath}/Kernel_linear.npz')

    #Oldroyd
    eps = None
    K = compute_kernel_matrix(X, X,'oldroyd', theta_matrix, eps,dx,dy)
    np.savez(f'{dspath}/Kernel_oldroyd.npz')

    #Poly
    eps = 3
    K = compute_kernel_matrix(X, X,'poly', None, eps,dx,dy)
    np.savez(f'{dspath}/Kernel_poly.npz')

    #RBF
    eps = 2
    K = compute_kernel_matrix(X, X,'rbf', None, eps,dx,dy)
    np.savez(f'{dspath}/Kernel_rbf.npz')

    #Cosine
    eps = None
    K = compute_kernel_matrix(X, X,'cosine', None, eps,dx,dy)
    np.savez(f'{dspath}/Kernel_cosine.npz')