import numpy as np
a = np.ones((2,2))
a@a
from utils import *
import Autoencoder
import torch
from torch.utils.data import DataLoader, Dataset
import time
import os
import glob

def calc_energy(X, Wi, beta, Re, dx = 0.0125, dy = None):
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
    area = dx * dy
    u = X[:,0::5]
    v = X[:,1::5]
    bxx = X[:,2::5]
    bxy = X[:,3::5]
    byy = X[:,4::5]

    kinetic = 0.5 * ((u**2 + v**2)*area).sum(0)
    txx = (bxx**2 + bxy**2 -1) * (1-beta)/Wi
    tyy = (bxy**2 + byy**2 -1) * (1-beta)/Wi

    elastic =  0.5 * ((txx + tyy)/(Re)*area).sum(0)


    total_energy = (kinetic+elastic) + (1-beta)/(Wi*Re)
    return elastic, kinetic, total_energy

def energy_loss(x,y,param, dx = 1/2**6):
    Wi = param[:,0].view((-1,1))
    beta= param[:,1].view((-1,1))
    ex = calc_energy(x, Wi, beta, 1, dx)
    ey = calc_energy(y, Wi, beta, 1, dx)

    loss = torch.nn.L1Loss(ex, ey)

    return loss

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

class FileDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()

        self.root_dir = root_dir
        self.filenames = glob.glob('*.pt', root_dir=root_dir)

    def __len__(self):
         return len(self.filenames)
    

    def __getitem__(self, index):
         data = torch.load(f'{self.root_dir}/{self.filenames[index]}')
         return data['tensor'].float(), torch.tensor(data['param']).float()
         
    
    
def get_min_max(dataset):
    min_v = dataset[0][0].amin(1)
    max_v = dataset[0][0].amax(1)
    for i in range(1,len(dataset)-1):
        min_v = torch.minimum(min_v, dataset[i][0].amin(1))
        max_v = torch.maximum(max_v, dataset[i][0].amax(1))
    return min_v.reshape((1,5,1)).clone().float(), max_v.reshape((1,5,1)).clone().float()


if __name__ == '__main__':

    torch.manual_seed(42) # reprodutibility
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    latent_dim = 3


    ## Data reading
    train_dataset = FileDataset('/container/fabio/npz_data/four_roll_train')
    test_dataset = FileDataset('/container/fabio/npz_data/four_roll_test')

    #normalize data inside autoencoder
    lower_bound,  upper_bound = get_min_max(train_dataset)
    lower_bound = lower_bound.to(device)
    upper_bound = upper_bound.to(device)

    # NN part
    learning_rate = 1e-4
    bs = 1000
    num_epochs = 5000

    autoencoder = Autoencoder.VAE_Transformer(n_input= train_dataset[0][0].shape[-1],latent_dim = latent_dim, num_params=3, max_in=upper_bound, min_in=lower_bound).to(device)


    train_loader = DataLoader(train_dataset, shuffle= True, batch_size=bs)
    test_loader =  DataLoader(test_dataset, shuffle= False, batch_size=len(test_dataset))
    loss_energy = True

    if loss_energy:
        def loss_fn(input:torch.Tensor, target:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, param:torch.tensor):
                # reconst_loss = torch.nn.MSELoss()(input, target)
                reconst_loss = energy_loss(input, target, param)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                kld_weight = 0.025
                return reconst_loss + kld_loss*kld_weight
    else:
        def loss_fn(input:torch.Tensor, target:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, param:torch.tensor = None):
                reconst_loss = torch.nn.MSELoss()(input, target)
                # reconst_loss = energy_loss(input, target, param)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                kld_weight = 0.025
                return reconst_loss + kld_loss*kld_weight
    optimizer = torch.optim.Adam(autoencoder.parameters(),lr = learning_rate)

    num_batches = len(train_loader)

    # Results directory
    pasta = f'/container/fabio/reduction-methods/ModelsTorch/VAETrans_4Roll_Latent_{latent_dim}_energy_{loss_energy}'
    os.makedirs(pasta, exist_ok=True)

    # Early stop
    best_vloss = 1_000_000
    last_loss = best_vloss
    patience = 0
    #training
    autoencoder.train(True)
    for e in range(num_epochs):
        if last_loss < best_vloss:
                        best_vloss = last_loss
                        torch.save({'optimizer_state_dict':optimizer.state_dict(), 'loss':loss, 'epoch':e}, f'{pasta}/optimizer_checkpoint.pt')
                        torch.save(autoencoder.state_dict(), f'{pasta}/best_autoencoder')
                        patience = 0
        else:
            patience += 1
        if patience > 50:
            autoencoder.load_state_dict(torch.load( f'{pasta}/best_autoencoder'))
            break

        cumm_loss = 0
        t = time.time()
        for data,param in train_loader:
            optimizer.zero_grad()
            # Use the context manager
            with ClearCache():
                data = data.to(device)
                param = param.to(device)
                reconst, mu, log_var = autoencoder(data,param)
                loss = loss_fn(data, reconst, mu, log_var, param)
                loss.backward()
                optimizer.step()

            cumm_loss += loss.item()
        t = time.time() - t
        last_loss = cumm_loss
        with torch.no_grad():
            for X_test,param in train_loader:
                X_test = X_test.to(device)
                reconst, mu, log_var = autoencoder(X_test,param.to(device))
                loss_test = loss_fn(X_test, reconst,mu, log_var)
        print(f'Epoch {e}: train loss: {cumm_loss:.4f}\ttest loss: {loss_test.item():.4f}', end='\t', flush=True)
        print(f'Exec. Time of epoch: {t:.3f}s({t/num_batches:.3f}s/batch)')

    print('\n\n')