import sys 
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))
import numpy as np
#for some reason, matplotlib crashes without these lines
a = np.zeros((5,5))
a@a
import Autoencoder
# from torchsummary import summary
# from Autoencoder import Autoencoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import calc_energy, np2torch
import glob
import time
# from importlib import reload
import argparse


device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)


def kernel(X:torch.Tensor, Y, theta, dx = 0.0125, dy = None):
    """
    Compute the total energy of a visco-elastic flow

    Parameters
    ----------
    X : array_like
        Simulation data
    Wi, beta, Re : float
                    Simulation paramters
    dx : float
        uniform mesh spacing

    Returns
    -------
    elastic : array
            The elastic energy on each snapshot of the input
    kinetic : array
            The kinect energy on each snapshot of the input
    total_energy: array
            The total energy on each snapshot of the input
    """
    if dy is None:
        dy = dx
    area = dx*dy * 0.5
    c = torch.ones((X.shape[0],5),device = X.device)
    c[:,2:] *= theta
    c[:,3] *= 2
    total_energy = torch.einsum('ijk, njk, ij -> in',X,Y,c)
    return torch.diag(total_energy) * area
def energy_loss(x,y,param, dx = 1/2**6):
    Wi = param[:,0].view((-1,1))
    beta= param[:,1].view((-1,1))
    theta = (1- beta) / Wi
    Kxx = kernel(x,x, theta, dx, dx)
    Kxy = kernel(x,y, theta, dx, dx)
    Kyy = kernel(y,y, theta, dx, dx)

    loss = torch.sqrt(Kxx - 2* Kxy + Kyy)
    
    return loss.mean()

def get_matrix(filename, dspath, ndata = 3000):
    #reads the file
    filename_no_ext = ".".join(filename.split('.')[:-1])
    f_split = filename_no_ext.split('_')
    Re = float(f_split[2].replace('Re',''))
    Wi = float(f_split[3].replace('Wi',''))
    beta = float(f_split[4].replace('beta',''))
    fields = np.load(f'{dspath}/{filename}', allow_pickle=True)["fields"].item()
    param =  np.repeat((Re,Wi,beta), ndata).reshape((3,-1)).T
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
    X =  T[:, -ndata:]
    return X.T, theta_sqrt, (Re, Wi, beta)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Loss', '-l', default='mse', type=str, help="Type of the loss ['mse' or 'energy']")
    parser.add_argument('--Latent', '-d', default=3, type=int, help="Latent dimension") 
    parser.add_argument('--Index', '-i', default=0, type=int, help="File Index") 

    dir_prefix = '/container/fabio'

    args = parser.parse_args()
    # dspath = '/home/fabio/npz_data/KPCA_4roll' # Four roll
    dspath = f'{dir_prefix}/npz_data/KPCA_4roll' # cavity
    file_ind = args.Index
    files = glob.glob('*.npz', root_dir=dspath)
    ntimes = 1000
    n_data = ntimes * len(files)
    X, sqrt_theta, param = get_matrix(files[file_ind], dspath, ntimes)

    Re = param[0]
    Wi = param[1]
    beta = param[2]
    theta_mult = sqrt_theta @ sqrt_theta.T
    theta = np.diag(theta_mult)[:,None]
    dx = dy = (np.pi)/32
    _, _, energy_X = calc_energy(X.T,Wi,beta, Re, dx, dy)

    X_torch = np2torch(X.T).float()
    train_dataset = TensorDataset(X_torch)
    train_loader = DataLoader(train_dataset, batch_size=1000)
    num_batches = len(train_loader)

    loss_energy = args.Loss.upper()

    mse_loss = torch.nn.MSELoss()
    if loss_energy == 'ENERGY':
            def loss_fn(input:torch.Tensor, target:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, param:torch.tensor, kld_weight = 0.0025):
                    # reconst_loss = torch.nn.MSELoss()(input, target)
                    reconst_loss = energy_loss(input, target, param)
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                    # kld_weight = 0.0025
                    return reconst_loss, kld_loss*kld_weight
    elif loss_energy == 'MSE':
            def loss_fn(input:torch.Tensor, target:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, param:torch.tensor = None, kld_weight = 0.0025):
                    reconst_loss = mse_loss(input, target)
                    # reconst_loss = energy_loss(input, target, param)
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                    # kld_weight = 0.0025
                    return reconst_loss, kld_loss*kld_weight
    elif loss_energy == 'BOTH':
            def loss_fn(input:torch.Tensor, target:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, param:torch.tensor, kld_weight = 0.0025):
                    # reconst_loss = torch.nn.MSELoss()(input, target)
                    reconst_loss = energy_loss(input, target, param) + mse_loss(input, target)
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                    # kld_weight = 0.0025
                    return reconst_loss, kld_loss*kld_weight
    else:
            raise Exception("Invalid loss function. Values are ['ENERGY', 'MSE', 'BOTH']")
    
    
    min_in = X_torch.amin(dim=2).amin(0).reshape((1,5,1)).float()
    max_in = X_torch.amax(dim=2).amax(0).reshape((1,5,1)).float()

    latent_dim = args.Latent

    num_epochs = 5000
    autoencoder = Autoencoder.VariationalAutoencoderModule(X_torch.shape[-1], latent_dim, max_in, min_in).to(device)

    learning_rate = 1e-4
    kl_weight = 1e-3
    optimizer = torch.optim.Adam(autoencoder.parameters(),lr = learning_rate)

    model_name  = files[file_ind].split('.')[0]
    pasta = f'{dir_prefix}/reduction-methods/ModelsTorch/VAE_Latent_{latent_dim}_energy_{loss_energy}_{model_name}'
    os.makedirs(pasta, exist_ok=True)

    # Early stop
    best_vloss = 1_000_000
    last_loss = best_vloss
    patience = 0
    param  = torch.tensor([[Wi, beta]]).float().to(device)

    for e in range(num_epochs):
        if last_loss < best_vloss:
                        best_vloss = last_loss
                        torch.save({'optimizer_state_dict':optimizer.state_dict(), 'loss':loss, 'epoch':e}, f'{pasta}/optimizer_checkpoint.pt')
                        torch.save(autoencoder.state_dict(), f'{pasta}/best_autoencoder')
                        patience = 0
        else:
                patience +=1

        if patience > 100:
            autoencoder.load_state_dict(torch.load( f'{pasta}/best_autoencoder'))
            break

        cumm_loss = 0
        cumm_loss_rec = 0
        cumm_loss_kld = 0
        cumm_loss_pred = 0
        t = time.time()
        autoencoder.train(True)
        for data_list in train_loader:
            data = data_list[0]
            optimizer.zero_grad()
            # Use the context manager
            # with ClearCache():
            data = data.to(device)
            
            code, mu, log_var = autoencoder.encode(data)
            reconst = autoencoder.decode(code)
            reconst_loss, kdl_loss = loss_fn(data, reconst, mu, log_var, param, kl_weight)

            loss = reconst_loss + kdl_loss 
            loss.backward()
            optimizer.step()

            cumm_loss += loss.item()
            cumm_loss_rec += reconst_loss.item()
            cumm_loss_kld += kdl_loss.item()
        t = time.time() - t
        last_loss = cumm_loss
        with torch.no_grad():
            autoencoder.eval()
            for X_test in train_loader:
                X_test = X_test[0].to(device)

                code, mu, log_var = autoencoder.encode(X_test)
                reconst = autoencoder.decode(code)
                loss_rec_test, loss_kld_test = loss_fn(X_test, reconst,mu, log_var, param, kl_weight)
                loss_test = loss_rec_test + loss_kld_test
        print(f'Epoch {e}: train loss: {cumm_loss:.4f}\ttest loss: {loss_test.item():.4f}\tExec. Time of epoch: {t:.3f}s({t/num_batches:.3f}s/batch)', flush=True)
        print(f'Reconst loss train: {cumm_loss_rec:.4f}, KLD loss train: {cumm_loss_kld:.4f}', flush=True)
        print(f'Reconst loss test: {loss_rec_test.item():.4f}, KLD loss test: {loss_kld_test.item():.4f}', flush=True)
        print(flush=True)

    print('\n\n')