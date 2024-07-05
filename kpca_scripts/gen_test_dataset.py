import sys 
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
import numpy as np
from KPCA import *
import glob
import os.path as path
import torch

DX = 1/(2**6)
PCA_KWD = {'kernel_type':'linear', 'theta':None, 'eps':None, 'dx' : DX, 'dy' : DX}
OLD_KWD = {'kernel_type':'oldroyd', 'theta':None, 'eps':None, 'dx' : DX, 'dy' : DX}
POL_KWD = {'kernel_type':'poly', 'theta':None, 'eps':3, 'dx' : DX, 'dy' : DX}
RBF_KWD = {'kernel_type':'rbf', 'theta':None, 'eps':1, 'dx' : DX, 'dy' : DX}
COS_KWD = {'kernel_type':'cosine', 'theta':None, 'eps':None, 'dx' : DX, 'dy' : DX}

def get_matrix(filename, dspath = '/home/fabio/npz_data/KPCA_4roll'):
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
    return X.T, theta_sqrt, param

if __name__ == '__main__':
    dspath = '/home/fabio/npz_data/KPCA_4roll'
    srcpath = '/home/fabio/npz_data/KPCA_test'
    files_test = glob.glob('4_roll6*.npz', root_dir=srcpath)
    dir_reconst = '/home/fabio/npz_data/Kernel_dataset_test/Kernel_reconstruction'
    dataset_test = '/home/fabio/npz_data/Kernel_dataset_test'
    
    npoints = 4096*5
    nfiles = 12
    datset_matrix = np.memmap(f'{dspath}/dataset.dat', mode='r', shape=(3000, npoints*nfiles))
    datset_theta = np.memmap(f'{dspath}/dataset_theta.dat', mode='r', shape=(3000*nfiles,1))

    k_args = [
        OLD_KWD,
        PCA_KWD,
        POL_KWD,
        RBF_KWD,
        COS_KWD
    ]
    os.makedirs(dir_reconst, exist_ok=True)
    for k in k_args:
        os.makedirs(f'{dataset_test}/Kernel_{k["kernel_type"]}', exist_ok=True)

    reduction = np.zeros((20,3000))
    for i in range(len(files_test)):
        X1, t1, param = get_matrix(files_test[i], srcpath)
        X_data = (X1.T).reshape((64*64,5,-1))
        X_data = np.moveaxis(X_data,[0,2],[2,0]) # (Nx, Nc, Nt) -> (Nt, Nc, Nx)
        _, Wi, beta = param[:, 0]
        # convert data
        X_torch = torch.from_numpy(X_data)
        for j,Xd in enumerate(X_torch):
            t = 0.1 * j
            rec_obj = {
                'y': Xd.clone(),
                'param':[Wi, beta, t]
                }
            torch.save(rec_obj,f'{dir_reconst}/data_{3000*i+j:06d}_Wi{Wi:g}_beta{beta:g}_t{t:g}.pt')
        for k_opt in k_args:
            U_data = np.load(f'{dspath}/U_fit_{k_opt["kernel_type"]}.npz', allow_pickle=True)
            mean = U_data['mean']
            U_fit = U_data['U']
            print(i, k_opt['kernel_type'])
            if k_opt['kernel_type'] == 'oldroyd':
                k_opt['theta'] = t1 @ datset_theta.T
            K = compute_kernel_matrix(X1, datset_matrix,**k_opt)

            x_kpca = (K - mean)@U_fit
            for j,xi_np in enumerate(x_kpca):
                t = 0.1 * j
                count = 3000*i +j
                xi = torch.from_numpy(xi_np[:])
                save_obj = {
                    'x':xi.clone(),
                    'y_code': count,
                }
                torch.save(save_obj,f'{dataset_test}/data_{count:06d}_Wi{Wi:g}_beta{beta:g}_t{t:g}.pt')
