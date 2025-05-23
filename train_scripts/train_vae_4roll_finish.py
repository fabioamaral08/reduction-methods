import sys
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))

import numpy as np
a = np.ones((2,2))
a@a
import torch
from torch.utils.data import DataLoader, Dataset
import time
import glob
from typing import Iterator, List
from utils import *
import Autoencoder
import argparse

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

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

class FileDataset(Dataset):
    def __init__(self, root_dir, take_time = True):
        super().__init__()

        self.root_dir = root_dir
        self.take_time = take_time
        self.filenames = glob.glob('*.pt', root_dir=root_dir)
        self.filenames.sort()
        self.cases = []
        for file in self.filenames:
            sfile = file.split('_')
            Wi = float(sfile[2].replace('Wi',''))
            beta = float(sfile[3].replace('beta',''))

            if (Wi, beta) not in self.cases:
                self.cases.append((Wi, beta))

    def __len__(self):
         return len(self.filenames)
    

    def __getitem__(self, index):
        while isinstance(index, list):
            index = index[0]
        data = torch.load(f'{self.root_dir}/{self.filenames[index]}')
        if self.take_time:
            return data['tensor'].float(), torch.tensor(data['param']).float()
        return data['tensor'].float(), torch.tensor(data['param'][:-1]).float()
         
class CaseSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data, cases, root_dir) -> None:
        self.data = data
        self.cases = cases
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for Wi, beta in self.cases:
            case = glob.glob(f'*Wi{Wi:g}_beta{beta:g}_*.pt', root_dir=self.root_dir)
            yield from [self.data.index[x] for x in case]

class CaseBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, data, cases, root_dir, batch_size: int) -> None:
        self.batch_size = batch_size
        self.iter_list = []

        for Wi, beta in cases:
            case = glob.glob(f'*Wi{Wi:g}_beta{beta:g}_*.pt', root_dir=root_dir)
            nchunks = (len(case) + self.batch_size - 1) // self.batch_size
            files_indexes = torch.tensor([data.index(x) for x in case])
            
            #sort files
            times = [self.get_t(fname) for fname in case]
            idx = np.argsort(times)
            files_indexes = files_indexes[idx]
            for batch in np.split(files_indexes, nchunks):
                self.iter_list.append(batch.tolist())

    def __len__(self) -> int:
        return len(self.iter_list)

    def __iter__(self):
        yield from self.iter_list

    def get_t(self, filename):
        str_s = filename.split('_')[4].replace('t','')
        t = float(str_s)
        return t
    
def get_min_max(dataset):
    min_v = dataset[0][0].amin(1)
    max_v = dataset[0][0].amax(1)
    for i in range(1,len(dataset)-1):
        min_v = torch.minimum(min_v, dataset[i][0].amin(1))
        max_v = torch.maximum(max_v, dataset[i][0].amax(1))
    return min_v.reshape((1,5,1)).clone().float(), max_v.reshape((1,5,1)).clone().float()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Loss', '-l', default='mse', type=str, help="Type of the loss ['mse' or 'energy']")
    parser.add_argument('--Beta', '-b', default=0.0025, type=float, help="KL divergence weight") 
    parser.add_argument('--Latent', '-d', default=3, type=int, help="Latent dimension") 
    parser.add_argument('--Pred', '-p', default=0, type=int, help="Use the predictor") 
    args = parser.parse_args()
    torch.manual_seed(42) # reprodutibility
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    latent_dim = args.Latent
    use_pred = args.Pred != 0

    ## Data reading
    train_dataset = FileDataset('/container/fabio/npz_data/four_roll_train_osc', take_time = False)
    test_dataset = FileDataset('/container/fabio/npz_data/four_roll_test_osc', take_time = False)

    #normalize data inside autoencoder
    lower_bound,  upper_bound = get_min_max(train_dataset)
    lower_bound = lower_bound.to(device)
    upper_bound = upper_bound.to(device)

    # NN part
    learning_rate = 1e-4
    bs = 3000
    num_epochs = 5000

    autoencoder = Autoencoder.ParametricVAEModule(n_input= train_dataset[0][0].shape[-1],latent_dim = latent_dim, num_params=2, max_in=upper_bound, min_in=lower_bound, pred=use_pred).to(device)

    # sampler = CaseSampler(train_dataset.filenames, train_dataset.cases, train_dataset.root_dir)
    batch_sampler_train = CaseBatchSampler(train_dataset.filenames, train_dataset.cases, train_dataset.root_dir, bs)
    batch_sampler_test = CaseBatchSampler(test_dataset.filenames, test_dataset.cases, test_dataset.root_dir, bs)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=0)
    test_loader =  DataLoader(test_dataset, batch_sampler=batch_sampler_test)
    loss_energy = args.Loss.upper() == 'ENERGY'
    mse_loss = torch.nn.MSELoss()
    if loss_energy:
        def loss_fn(input:torch.Tensor, target:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, param:torch.tensor, kld_weight = 0.0025):
                # reconst_loss = torch.nn.MSELoss()(input, target)
                reconst_loss = energy_loss(input, target, param)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                # kld_weight = 0.0025
                return reconst_loss, kld_loss*kld_weight
    else:
        def loss_fn(input:torch.Tensor, target:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, param:torch.tensor = None, kld_weight = 0.0025):
                reconst_loss = mse_loss(input, target)
                # reconst_loss = energy_loss(input, target, param)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                # kld_weight = 0.0025
                return reconst_loss, kld_loss*kld_weight
    optimizer = torch.optim.Adam(autoencoder.parameters(),lr = learning_rate)

    num_batches = len(train_loader)

    # Results directory
    pasta = f'/container/fabio/reduction-methods/ModelsTorch/VAE_4RollOSC_Latent_{latent_dim}_energy_{loss_energy}_beta_{args.Beta:g}'
    os.makedirs(pasta, exist_ok=True)
    checkpoint = torch.load(f'{pasta}/optimizer_checkpoint.pt')
    autoencoder.load_state_dict(torch.load(f'{pasta}/best_autoencoder'))

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # Early stop
    best_vloss = 1_000_000
    last_loss = best_vloss

    #training
    kl_weight = args.Beta
    for e in range(epoch,num_epochs):
        if last_loss < best_vloss:
                        best_vloss = last_loss
                        torch.save({'optimizer_state_dict':optimizer.state_dict(), 'loss':loss, 'epoch':e}, f'{pasta}/optimizer_checkpoint.pt')
                        torch.save(autoencoder.state_dict(), f'{pasta}/best_autoencoder')



        cumm_loss = 0
        cumm_loss_rec = 0
        cumm_loss_kld = 0
        cumm_loss_pred = 0
        t = time.time()
        autoencoder.train(True)
        for data,param in train_loader:

            optimizer.zero_grad()
            # Use the context manager
            # with ClearCache():
            data = data.to(device)
            param = param.to(device)
            code, mu, log_var = autoencoder.encode(data,param)
            reconst = autoencoder.decode(code,param)
            reconst_loss, kdl_loss = loss_fn(data, reconst, mu, log_var, param, kl_weight)
            if use_pred:
                inpt_pred = code[:-1]
                out_pred = code[1:]
                forecast = autoencoder.predictor(inpt_pred)
                pred_loss = mse_loss(out_pred, forecast)
                cumm_loss_pred += pred_loss.item()
            else:
                 pred_loss = 0.0
            loss = reconst_loss + kdl_loss + pred_loss
            loss.backward()
            optimizer.step()

            cumm_loss += loss.item()
            cumm_loss_rec += reconst_loss.item()
            cumm_loss_kld += kdl_loss.item()
        t = time.time() - t
        last_loss = cumm_loss
        with torch.no_grad():
            autoencoder.eval()
            for X_test,param in train_loader:
                X_test = X_test.to(device)
                param = param.to(device)

                code, mu, log_var = autoencoder.encode(X_test,param)
                reconst = autoencoder.decode(code,param)
                loss_rec_test, loss_kld_test = loss_fn(X_test, reconst,mu, log_var, param, kl_weight)
                if use_pred:
                    inpt_pred = code[:-1]
                    out_pred = code[1:]
                    forecast = autoencoder.predictor(inpt_pred)
                    loss_pred_test = mse_loss(out_pred, forecast).item()
                else:
                     loss_pred_test=0.0
                loss_test = loss_pred_test + loss_rec_test + loss_kld_test
        print(f'({args.Loss.upper()})Epoch {e}: train loss: {cumm_loss:.4f}\ttest loss: {loss_test.item():.4f}\tExec. Time of epoch: {t:.3f}s({t/num_batches:.3f}s/batch)', flush=True)
        print(f'Reconst loss train: {cumm_loss_rec:.4f}, KLD loss train: {cumm_loss_kld:.4f}, pred loss train: {cumm_loss_pred:.4f}', flush=True)
        print(f'Reconst loss test: {loss_rec_test.item():.4f}, KLD loss test: {loss_kld_test.item():.4f}, pred loss test: {loss_pred_test:.4f}', flush=True)
        print(flush=True)

    print('\n\n')