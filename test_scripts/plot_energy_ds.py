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
import argparse
import matplotlib.pyplot as plt

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

def calc_dkl(X,param, ae):
    with torch.no_grad():
        _, mu, log_var = ae.encode(X, param)
        return -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 0)


BETA_IND = {
    0.11111:0,
    0.30:1,
    0.40:2,
    0.50:3,
    0.75:4,
    0.90:5,
}
WI_IND = {
    2.0:0,
    2.8:1,
    3.0:2,
    3.2:3,
    3.5:4,
    3.6:5,
    3.8:6,
    4.0:7,
}
WI_IND_TEST = {
    2.25:0,
    2.50:1,
    2.65:2,
    3.10:3,
    3.30:4,
    3.35:5,
    3.40:6,
    3.45:7,
    4.20:7,
    4.35:7,
}
if __name__ == '__main__':

    ## Data reading
    train_dataset = FileDataset('/node/fabio/npz_data/four_roll_train_osc', take_time = False)
    test_dataset = FileDataset('/node/fabio/npz_data/four_roll_test_osc', take_time = False)

    # #normalize data inside autoencoder
    # lower_bound,  upper_bound = get_min_max(train_dataset)
    # lower_bound = lower_bound.to(device)
    # upper_bound = upper_bound.to(device)

    bs = 3000

    # sampler = CaseSampler(train_dataset.filenames, train_dataset.cases, train_dataset.root_dir)
    batch_sampler_train = CaseBatchSampler(train_dataset.filenames, train_dataset.cases, train_dataset.root_dir, bs)
    batch_sampler_test = CaseBatchSampler(test_dataset.filenames, test_dataset.cases, test_dataset.root_dir, bs)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=0)
    test_loader =  DataLoader(test_dataset, batch_sampler=batch_sampler_test)


    dx = 2 * np.pi / 2**6
    Re = 1

    f_train, ax_train = plt.subplots(8,6, figsize = (50,30))
    for i in range(8):
        ax_train[i,0].set_ylabel(f'Wi = {list(WI_IND.keys())[i]}')
        for j in range(6):
            if i == 0:
                ax_train[-1,j].set_xlabel(f'$\\beta$ = {list(BETA_IND.keys())[i]}')
            ax_train[i,j].set_xticks([])
            ax_train[i,j].set_yticks([])
    for data,param in train_loader:
        data = data.cpu()
        param = param.cpu()
        with torch.no_grad():
            X = torch2np(data)
            Wi = param[0,0].item()
            b = param[0,1].item()
            Wi = float(f'{Wi:g}')
            b = float(f'{b:g}')

        # Energy From data
        _, _, total = calc_energy(X,Wi,b,Re, dx = dx)
        i = WI_IND[Wi]
        j = BETA_IND[b]
        ax_train[i,j].plot(total, label = 'SIMULATION',color='k', lw = 1)

    f_train.savefig('/node/fabio/reduction-methods/test_scripts/Results/Energy_Train.png')

    f_test, ax_test = plt.subplots(10,6, figsize = (50,30))
    for i in range(10):
        ax_test[i,0].set_ylabel(f'Wi = {list(WI_IND_TEST.keys())[i]}')
        for j in range(6):
            if i == 0:
                ax_test[-1,j].set_xlabel(f'$\\beta$ = {list(BETA_IND.keys())[i]}')
            ax_test[i,j].set_xticks([])
            ax_test[i,j].set_yticks([])
    for data,param in test_loader:
        data = data.cpu()
        param = param.cpu()
        with torch.no_grad():
            X = torch2np(data)
            Wi = param[0,0].item()
            b = param[0,1].item()
            Wi = float(f'{Wi:g}')
            b = float(f'{b:g}')

        # Energy From data
        _, _, total = calc_energy(X,Wi,b,Re, dx = dx)
        i = WI_IND_TEST[Wi]
        j = BETA_IND[b]
        ax_test[i,j].plot(total, label = 'SIMULATION',color='k', lw = 1)

    f_test.savefig('/node/fabio/reduction-methods/test_scripts/Results/Energy_Test.png')