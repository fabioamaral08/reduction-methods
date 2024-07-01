import sys 
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
import numpy as np
import glob
import torch

def get_matrix(filename):
    #reads the file
    f_split = filename.split('_')
    Wi = float(f_split[3].replace('Wi',''))
    beta = float(f_split[4].replace('beta',''))
    fields = np.load(f'{dspath}/{filename}', allow_pickle=True)["fields"].item()
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

    X =  T[:,:3000]
    return X, Wi, beta

if __name__ == '__main__':
    dspath = '/home/fabio/npz_data/KPCA_4roll'
    files = glob.glob('4_roll6*.npz', root_dir=dspath)
    X = []
    P = []
    n_data = 3000 * len(files)

    dirsave = [   
        f'{dspath}/Kernel_train_oldroyd',
        f'{dspath}/Kernel_train_linear',
        f'{dspath}/Kernel_train_poly',
        f'{dspath}/Kernel_train_rbf',
        f'{dspath}/Kernel_train_cosine'
        ]

    for d in dirsave:
        os.makedirs(d, exist_ok=True)
    n_components = 20
    kpca_files = [   
        f'oldroyd',
        f'linear',
        f'poly',
        f'rbf',
        f'cosine'
        ]
    matrixes = [np.memmap(f'{dspath}/X_{x}.npz',dtype='float32', mode='w+', shape=(n_data,n_components)) for x in kpca_files]
    count = 0
    for i in range(len(files)):
        X1, Wi, beta = get_matrix(files[i])
        X_data = X1.reshape((64*64,5,-1))
        X_data = np.moveaxis(X_data,[0,2],[2,0]) # (Nx, Nc, Nt) -> (Nt, Nc, Nx)

        # convert data
        X_torch = torch.from_numpy(X_data)
        for i,Xd in enumerate(X_torch):
            t = 0.1 * i
            for X, dataset_train in zip(matrixes, dirsave):
                xi = torch.from_numpy(X[count])

                save_obj = {
                    'x':xi.clone(),
                    'y': Xd.clone(),
                    'param':[Wi, beta, t]
                }
                torch.save(save_obj,f'{dataset_train}/data_{count:06d}_Wi{Wi:g}_beta{beta:g}_t{t:g}.pt')
            count +=1
        
