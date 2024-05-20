import sys 
sys.path.append('../src') 
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

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


if __name__ == '__main__':

    torch.manual_seed(42) # reprodutibility
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    latent_dim = 3
    parameters = [
        #   (0.01,2.5,0.11111),
        #   (0.01,3.5,0.11111),
        #   (0.01,5.0,0.33333),
        #   (0.01,7.0,0.11111),
          (0.01,7.5,0.33333),
    ]
    latent_dim = 3

    for params in parameters:
        ## Data reading
        # Parameters:
        Re ,Wi ,beta = params

        # type of simulation
        case = 'cross'
        #read file
        dirpath_data = '../EnergyReduction/npz_data'
        X, Xmean = get_data(Re,Wi,beta, case, n_data= -2, dir_path=dirpath_data)

        Nt = X.shape[1] # number of snapshots
        q = X.reshape((181,181,5,-1))


        cut = 13 # removes the influence of inflow/outflow

        X_data = strip_cross(q, cut)
        X_data = np.moveaxis(X_data,[0,2],[2,0]) # (Nx, Nc, Nt) -> (Nt, Nc, Nx)

        # convert data
        X_torch = torch.from_numpy(X_data)

        #normalize data inside autoencoder
        lower_bound = torch.from_numpy(X_data.min(axis = (0,2)).reshape((1,5,1))).float().to(device)
        upper_bound = torch.from_numpy(X_data.max(axis = (0,2)).reshape((1,5,1))).float().to(device)
        # X_torch = (X_torch - lower_bound)/(upper_bound - lower_bound)

        # NN part
        learning_rate = 1e-4
        bs = 100
        num_epochs = 5000

        autoencoder = Autoencoder.AutoencoderModule(n_input= X_torch.shape[-1], latent_dim = latent_dim, max_in=upper_bound, min_in=lower_bound).to(device)

        
        X_torch = X_torch.float().to(device)
        X_train = X_torch[:-100]
        X_test = X_torch[-100:]
        dataset = TensorDataset(X_train,X_train)
        loader = DataLoader(dataset, shuffle= True, batch_size=bs)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(),lr = learning_rate)

        num_batches = len(loader)

        # Results directory
        pasta = f'ModelsTorch/Dense_CrossPred_Latent_{latent_dim}/Re{Re:g}_Wi{Wi:g}_beta{beta:g}'
        os.makedirs(pasta, exist_ok=True)

        # Early stop
        best_vloss = 1_000_000
        last_loss = best_vloss
        patience = 0
        #training
        autoencoder.train(True)
        print(f'Starting: ({Re:g}, {Wi:g}, {beta:g})\n Shapes: {X_train.shape}\t{X_test.shape}')
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
                # Use the context manager
                with ClearCache():
                    data = data.to(device)
                    reconst = autoencoder(data)
                    loss = loss_fn(data, reconst)
                    loss.backward()
                    optimizer.step()

                cumm_loss += loss.item()
            t = time.time() - t
            last_loss = cumm_loss
            with torch.no_grad():
                reconst = autoencoder(X_test)
                loss_test = loss_fn(X_test, reconst)
            print(f'Epoch {e}: train loss: {cumm_loss:.4f}\ttest loss: {loss_test.item():.4f}', end='\t')
            print(f'Exec. Time of epoch: {t:.3f}s({t/num_batches:.3f}s/batch)')

        print('\n\n')