import numpy as np
a = np.ones((2,2))
a@a
from utils import *
import Autoencoder
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import time
import os

if __name__ == '__main__':

    latent_dim = 2
    ## Data reading

    # Parameters:
    Re = 1
    Wi = 5
    beta = 0.1
    # type of simulation
    case = '4roll'
    #read file
    X, Xmean = get_data(Re,Wi,beta, case, n_data= -2, dir_path='../EnergyReduction/npz_data')

    Nt = X.shape[1] # number of snapshots
    X_data = X.reshape((-1,5,Nt))
    X_data = np.moveaxis(X_data,[0,2],[2,0]) # (Nx, Nc, Nt) -> (Nt, Nc, Nx)

    # convert data
    X_torch = torch.from_numpy(X_data)

    #normalize data inside autoencoder
    lower_bound = X_data.min(axis = (0,2)).reshape((1,5,1))
    upper_bound = X_data.max(axis = (0,2)).reshape((1,5,1))
    # X_torch = (X_torch - lower_bound)/(upper_bound - lower_bound)

    # NN part
    learning_rate = 1e-4
    bs = 100
    num_epochs = 5000

    autoencoder = Autoencoder.AutoencoderModule(n_input= X_torch.shape[-1], latent_dim = latent_dim, max_in=upper_bound, min_in=lower_bound)

    dataset = TensorDataset(X_torch.float(),X_torch)
    loader = DataLoader(dataset, shuffle= True, batch_size=bs)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(),lr = learning_rate)

    num_batches = len(loader)

    # Results directory
    pasta = f'ModelsTorch/Dense_Latent_{latent_dim}'
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
        for data,_ in loader:
            optimizer.zero_grad()

            reconst = autoencoder(data)
            loss = loss_fn(data, reconst)
            loss.backward()
            optimizer.step()

            cumm_loss += loss.item()
        t = time.time() - t
        last_loss = cumm_loss
        print(f'Epoch {e}: running loss: {cumm_loss:.4f}')
        print(f'Exec. Time of epoch: {t:.3f}s({t/num_batches:.3f}s/batch)\n')

    torch.save(autoencoder,f'{pasta}/encoder_model')