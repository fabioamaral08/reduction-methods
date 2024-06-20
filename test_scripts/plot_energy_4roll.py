## Create a zip file with data for AE/VAE

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
from utils import *

def split_param(str):

    s = str.split('_')
    Re = s[2].replace('Re','')
    Wi = s[3].replace('Wi','')
    beta = s[4].replace('beta','').replace('.npz','')

    Re, Wi, beta = [float(x) for x in [Re, Wi, beta]]
    return Re, Wi, beta


def get_data(filename, root_dir):
    #reads the file
    fields = np.load(f'{root_dir}/{filename}', allow_pickle=True)["fields"].item()

    #Extract the fields
    u = fields["vel-u"]
    v = fields["vel-v"]
    Bxx = fields["Bxx"]
    Bxy = fields["Bxy"]
    Byy = fields["Byy"]
    q = np.stack((u,v,Bxx, Bxy, Byy), axis=-1)

    # q[:65,:65] = 0
    # q[:65,-65:] = 0
    # q[-65:,:65] = 0
    # q[-65:,-65:] = 0

    # reshape for the expected code format
    TU = q[:,:,:,0].reshape((q.shape[0]**2, q.shape[2]))
    TV = q[:,:,:,1].reshape((q.shape[0]**2, q.shape[2]))
    T11 = q[:,:,:,2].reshape((q.shape[0]**2, q.shape[2]))
    T12 = q[:,:,:,3].reshape((q.shape[0]**2, q.shape[2]))
    T22 = q[:,:,:,4].reshape((q.shape[0]**2, q.shape[2]))
    T = np.concatenate((TU, TV, T11,T12,T22), axis=1).reshape(-1, q.shape[2]) # by column axis=1(intercal..), by row axis=0


    return T[:,1:-1]

if __name__ == '__main__':

    # root = '/home/hugo/codes_basilisk/four_roll/post_proc'
    root = '/home/hugo/CodeMestrado_Cavity/post_proc'
    # filetemplate = '4_roll6_Re1_*_dataset.npz' # crossTurb_data_Re*_Wi*_beta*.npz
    # filetemplate = '4_roll6_Re1_*.npz' # crossTurb_data_Re*_Wi*_beta*.npz
    filetemplate = 'cross*.npz' # crossTurb_data_Re*_Wi*_beta*.npz
    filelist = glob.glob(filetemplate, root_dir=root)
    filelist = np.unique([x for x in filelist if x.count('a002') == 0])

    dataset_train = 'four_roll_train_osc_3k'
    # dataset_test = 'four_roll_test'

    # os.makedirs(dataset_train, exist_ok=True)
    # os.makedirs(dataset_test, exist_ok=True)

    cont_train = 0
    cont_test = 0
    cases = []
    dx = 1/2**6

    Wis = []
    betas = []
    for filecount, file in enumerate(filelist):
        Re, Wi, beta = split_param(file)
        # if Wi < 4.5: 
        #     continue
        if (Wi, beta) in cases:
            continue
        cases.append((Wi, beta))

    cases = np.array(cases)

    Wis = np.unique(cases[:,0])
    betas = np.unique(cases[:,1])

    nrow = len(Wis)
    ncol = len(betas)
    f_train, ax_train = plt.subplots(nrow,ncol, figsize = (ncol*6,ncol*6))
    print(nrow, ncol)
    print(Wis)
    print(betas)
    for i in range(nrow):
        ax_train[i,0].set_ylabel(Wis[i])
        for j in range(ncol):
            if i == 0:
                ax_train[-1,j].set_xlabel(f'$\\beta$ = {betas[j]}')
            ax_train[i,j].set_xticks([])
            ax_train[i,j].set_yticks([])
            
    for filecount, file in enumerate(filelist):
        print("Converting step: %g out of %g" % (filecount, len(filelist)))
        Re, Wi, beta = split_param(file)
        print(Re, Wi, beta, '\n')

        # if Wi < 4.5: 
        #     continue

        X = get_data(file, root)
        # print(Re, Wi, beta)
        # X_data = X.reshape((64*64,5,-1))
        Nt = X.shape[-1]
        print(Nt)
        # convert data
        # if Nt < 3000:
        #     print(f'{Re}, {Wi}, {beta} - Not enough data')
        #     continue
        # X = X[...,:3000]
        # Energy From data
        _, _, total = calc_energy(X,Wi,beta,Re, dx = dx)

        # print(np.argwhere(Wi == Wis))
        # print(np.argwhere(beta == betas))
        i = np.argwhere(Wi == Wis).flatten().item()
        j = np.argwhere(beta == betas).flatten().item()
        print(i,j)
        ax_train[i,j].plot(total, lw = 1)


    f_train.tight_layout()
    f_train.savefig('/home/fabio/Cross_slot.png')