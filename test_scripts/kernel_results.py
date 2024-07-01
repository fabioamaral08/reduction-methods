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
            return data['x'].float(), data['y'].float(), torch.tensor(data['param']).float()
        return data['x'].float(), data['y'].float(), torch.tensor(data['param'][:-1]).float()
         
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
        str_s = filename.split('_')[4].replace('t','').replace('.p','')
        t = float(str_s)
        return t
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Loss', '-l', default='mse', type=str, help="Type of the loss ['mse' or 'energy']")

    args = parser.parse_args()
    torch.manual_seed(42) # reprodutibility
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    latent_dim = 20
    use_pred = False

    ## Data reading
    
    # #normalize data inside autoencoder
    # lower_bound,  upper_bound = get_min_max(train_dataset)
    lower_bound = torch.zeros((1,5,1)).to(device)
    upper_bound = torch.ones((1,5,1)).to(device)

    bs = 3000
    loss_energy = args.Loss.upper() == 'ENERGY'

    # sampler = CaseSampler(train_dataset.filenames, train_dataset.cases, train_dataset.root_dir)
    # batch_sampler_test = CaseBatchSampler(test_dataset.filenames, test_dataset.cases, test_dataset.root_dir, bs)



    kernels = ['linear', 'oldroyd', 'poly', 'cosine', 'rbf']

    dx = 2 * np.pi / 2**6
    Re = 1
    train_dataset = FileDataset(f'/container/fabio/npz_data/Kernel_dataset/Kernel_train_oldroyd', take_time = False)
    autoencoder = Autoencoder.KernelDecoderModule(n_input= train_dataset[0][1].shape[-1],latent_dim = latent_dim, num_params=2, max_in=upper_bound, min_in=lower_bound,).to(device)
    for ker in kernels:
        pasta = f'/container/fabio/reduction-methods/ModelsTorch/Kernel_4RollOSC_Latent_{latent_dim}_energy_{loss_energy}_kernel_{ker}'

        train_dataset = FileDataset(f'/container/fabio/npz_data/Kernel_dataset/Kernel_train_{ker}', take_time = False)
        batch_sampler_train = CaseBatchSampler(train_dataset.filenames, train_dataset.cases, train_dataset.root_dir, bs)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=0)

        autoencoder.load_state_dict(torch.load(f'{pasta}/best_autoencoder',map_location=device))
        autoencoder.eval()
        for code, data,param in train_loader:
            data = data.to(device)
            code = code.to(device)
            param = param.to(device)
            with torch.no_grad():
                X = torch2np(data)
                Wi = param[0,0].item()
                b = param[0,1].item()
                if device_type == "cuda":
                    Wi_data = param[:,0].cpu().numpy()
                    beta_data = param[:,1].cpu().numpy()
                else:
                    Wi_data = param[:,0].numpy()
                    beta_data = param[:,1].numpy()

                theta_data = ((1-beta_data)/(Re * Wi_data))[:,None]

                X_ae_torch = autoencoder.decode(code, param)
                X_ae = torch2np(X_ae_torch)

            # Energy From data
            _, _, total = calc_energy(X,Wi,b,Re, dx = dx)

            # # Energy From Autoencoder
            _, _, total_mse = calc_energy(X_ae,Wi,b,Re,dx = dx)



            # Reconstruction Error:
            energy_norm_x = np.abs(total).sum()
            energy_err = np.abs(total - total_mse).sum()/energy_norm_x


            with open(f'/container/fabio/reduction-methods/test_scripts/Results/results_Kernel_4RollOSC_Latent_{latent_dim}_energy_{loss_energy}_Kernel_{ker}_train.txt', 'a+') as f:
                f.write(f'Wi: {Wi:g}, beta: {b:g}, theta: {theta_data[0].item():g}\n')
                f.write(f'Rel. energy error MSE: {energy_err:g}\n')

        # for data,param in test_loader:
        #     data = data.to(device)
        #     param = param.to(device)
        #     with torch.no_grad():
        #         X = torch2np(data)
        #         Wi = param[0,0].item()
        #         b = param[0,1].item()

        #         if device_type == "cuda":
        #             Wi_data = param[:,0].cpu().numpy()
        #             beta_data = param[:,1].cpu().numpy()
        #         else:
        #             Wi_data = param[:,0].numpy()
        #             beta_data = param[:,1].numpy()

        #         theta_data = ((1-beta_data)/(Re * Wi_data))[:,None]

        #         z,mu, log_var = autoencoder.encode(data, param)
        #         X_ae_torch = autoencoder.decode(z, param)
        #         X_ae = torch2np(X_ae_torch)

        #     # Energy From data
        #     _, _, total = calc_energy(X,Wi,b,Re, dx = dx)

        #     # # Energy From Autoencoder
        #     _, _, total_mse = calc_energy(X_ae,Wi,b,Re,dx = dx)



        #     # Reconstruction Error:
        #     energy_norm_x = np.abs(total).sum()
        #     energy_err = np.abs(total - total_mse).sum()/energy_norm_x

        #     #DKL:
        #     dkl = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 0)
        #     with open(f'/container/fabio/reduction-methods/test_scripts/Results/results_VAE_4RollOSC_Latent_{latent_dim}_energy_{loss_energy}_beta_{beta:g}_test.txt', 'a+') as f:
        #         f.write(f'Wi: {Wi:g}, beta: {b:g}, theta: {theta_data[0].item():g}\n')
        #         f.write(f'Rel. energy error MSE: {energy_err:g}\n')
        #         f.write(f'KL Divergence: {[f"{x:g}" for x in dkl]}\n\n')