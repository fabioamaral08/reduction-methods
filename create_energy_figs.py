import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from utils import get_data, calc_energy

if __name__ == '__main__':


    dirpath = '../EnergyReduction/npz_data/'
    plotdir = 'EnergyPlots/4roll/'
    filelist = glob.glob(dirpath + '4_roll6_Re*_Wi*_beta*_dataset.npz')
    os.makedirs(plotdir, exist_ok=True)

    dx = 1/(2**6)
    for file in filelist:
        s_file = file.split('_')
        Re_str = s_file[-4].replace('Re','')
        Wi_str = s_file[-3].replace('Wi','')
        beta_str = s_file[-2].replace('beta','')

        Re = float(Re_str)
        Wi = float(Wi_str)
        beta = float(beta_str)

        alpha = (1-beta)/(Re*Wi)

        X, Xmean = get_data(Re,Wi,beta, '4roll', n_data= -2, dir_path=dirpath)
        _, _, total = calc_energy(X,Wi,beta,Re, dx =dx, dy = dx)

        plt.plot(total, label = 'SIMULATION')
        plt.savefig(f'{plotdir}energy_4roll_Re{Re:g}_Wi{Wi:g}_beta{beta:g}_.png')
        plt.close()