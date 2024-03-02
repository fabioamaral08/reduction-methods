import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from utils import get_data, calc_energy, get_mesh_vtk

if __name__ == '__main__':


    dirpath = '/home/hugo/CodeMestrado_Cavity/post_proc'
    plotdir = 'EnergyPlots/cross/'
    filelist = glob.glob(dirpath + '/crossTurb_data_*.npz')
    os.makedirs(plotdir, exist_ok=True)

    # Mesh information:
    vtk_file = '../EnergyReduction/npz_data/Dados-N0.vtk'
    x, y = get_mesh_vtk(vtk_file)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    xc = (x[1:] + x[:-1])/2
    yc = (y[1:] + y[:-1])/2
    X, Y = np.meshgrid(xc,yc)
    DX, DY = np.meshgrid(dx,dy)
    for file in filelist:
        s_file = file.split('_')
        Re_str = s_file[-3].replace('Re','')
        Wi_str = s_file[-2].replace('Wi','')
        beta_str = s_file[-1].replace('beta','').replace('.npz','')

        Re = float(Re_str)
        Wi = float(Wi_str)
        beta = float(beta_str)

        alpha = (1-beta)/(Re*Wi)

        X, Xmean = get_data(Re,Wi,beta, 'cross', n_data= -2, dir_path=dirpath)
        _, _, total = calc_energy(X,Wi,beta,Re, dx =DX.reshape((-1,1)), dy = DY.reshape((-1,1)))

        plt.plot(total, label = 'SIMULATION')
        plt.savefig(f'{plotdir}energy_cross_Re{Re:g}_Wi{Wi:g}_beta{beta:g}_.png')
        plt.close()