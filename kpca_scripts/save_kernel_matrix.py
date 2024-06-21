import sys 
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
import numpy as np
from KPCA import *
import glob

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
    files = glob.glob('4_roll6*.npz', root_dir=dspath)
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
    k_args = [
        OLD_KWD,
        PCA_KWD,
        POL_KWD,
        RBF_KWD,
        COS_KWD
    ]

    matrixes = [np.memmap(x,dtype='float32', mode='w+', shape=(n_data,n_data)) for x in mat_files]
    
    for i in range(len(files)):
        X1, t1 = get_matrix(files[i])
        for j in range(i,len(files)):
            X2, t2 = get_matrix(files[j])

            for k_opt,M in zip(k_args, matrixes):
                print(i,j,k['kernel_type'])
                if k_opt['kernel_type'] == 'oldroyd':
                    k_opt['theta'] = t1 @ t2.T

                    print(k_opt['theta'].shape)
                K = compute_kernel_matrix(X1, X2,**k_opt)
                M[i*3000:(i+1)*3000, j*3000:(j+1):3000] = K
                M[j*3000:(j+1)*3000, i*3000:(i+1):3000] = K.T

                M.flush()
