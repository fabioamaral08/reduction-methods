import numpy as np

__all__ = ['get_data', 'calc_energy', 'tau2conf', 'newton_B', 'get_data_toy', 'get_mesh_vtk']

def get_data(Re, Wi, beta = 0.5, case = 'cavity_ref', n_data = 100, from_end= False, eps = None, dir_path = 'npz_data'):
    """
    Reads a file that contains the data of the simulation given the paramters.

    Parameters
    ----------
    Re, Wi, beta : float
                    Fluid paramters
    case : str, optional (default: 'cavity_ref')
            simulation case
    n_data: int, optional (default: 100)
            number of snapshots
    from_end: bool, optional (default: False)
            Which snapshots takes, If True it will select the last the n_data. If False it will select the n_data first data, skipping the first data (noisy transistion)
    eps : float, optional (default: None)
            Extra parameter for case 'giesekus', 'ptt' and 'fene-p'
    dir_path : str, optional (default: 'npz_data')
                Directory path of the data

    Returns
    -------
    X : np.array
        The data
    Xmean : np.array
            the temporal mean of the data 
    """

    #find the filename
    if case == 'cross': # Cross-SLot Geometry
        in_filename = f"crossTurb_data_Re0.01_Wi5_beta{beta:g}.npz" # u,v,B
    elif case == '4roll': # 4-roll geometry
        in_filename = f"4_roll6_Re1_Wi{Wi:g}_beta{beta:g}_dataset.npz"
    elif case == 'giesekus': # cavity - Giesekus Fluid
        in_filename = f"cavity_data_Re{Re:g}_Wi{Wi:g}_beta{beta:g}.npz" # u,v,B
    elif case == 'cavity': #cavity - Oldroyd fluid
        in_filename = f"cavitytransU_data_Re{Re:g}_Wi{Wi:g}_beta{beta:g}.npz" # u,v,B
    elif case == 'cavity_ref': #refined mesh - Oldroyd fluid
        in_filename = f"cavitytransU80_data_Re{Re:g}_Wi{Wi:g}_beta{beta:g}.npz" # u,v,B
    elif case == 'giesekus':
        in_filename = f"cavitytransU80_data_Re{Re:g}_Wi{Wi:g}_beta{beta:g}a{eps:g}.npz" # u,v,B
    elif case == 'ptt':
        in_filename = f"cavitytransU80_data_Re{Re:g}_Wi{Wi:g}_beta{beta:g}e{eps:g}.npz" # u,v,B
    elif case == 'fene-p':
        in_filename = f"cavitytransU80_data_Re{Re:g}_Wi{Wi:g}_beta{beta:g}l{eps:g}.npz" # u,v,B
    else:
        raise ValueError("Invalid case.")
    
    #reads the file
    fields = np.load(f'{dir_path}/{in_filename}', allow_pickle=True)["fields"].item()

    #Extract the fields
    u = fields["vel-u"]
    v = fields["vel-v"]
    Bxx = fields["Bxx"]
    Bxy = fields["Bxy"]
    Byy = fields["Byy"]
    q = np.stack((u,v,Bxx, Bxy, Byy), axis=-1)

    if case == 'cross': # Consider only the center of the channel
        q = q[65:116,65:116]

    # reshape for the expected code format
    TU = q[:,:,:,0].reshape((q.shape[0]**2, q.shape[2]))
    TV = q[:,:,:,1].reshape((q.shape[0]**2, q.shape[2]))
    T11 = q[:,:,:,2].reshape((q.shape[0]**2, q.shape[2]))
    T12 = q[:,:,:,3].reshape((q.shape[0]**2, q.shape[2]))
    T22 = q[:,:,:,4].reshape((q.shape[0]**2, q.shape[2]))
    T = np.concatenate((TU, TV, T11,T12,T22), axis=1).reshape(-1, q.shape[2]) # by column axis=1(intercal..), by row axis=0

    # Select the number of data
    if from_end:
        X = T[:,-n_data:-1]
    else:
        X = T[:,1:n_data+1]
    
    # Compute the temporal mean
    Xmean = X.mean(1).reshape(-1,1)
    return X, Xmean

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
    u = X[0::5]
    v = X[1::5]
    bxx = X[2::5]
    bxy = X[3::5]
    byy = X[4::5]

    kinetic = ((u**2 + v**2)*area * .5).sum(0)
    txx = (bxx**2 + bxy**2 -1) * (1-beta)/Wi
    tyy = (bxy**2 + byy**2 -1) * (1-beta)/Wi

    elastic = ((txx + tyy)/(Re)*area * .5).sum(0)


    total_energy = (kinetic+elastic) + (1-beta)/(Wi*Re)
    return elastic, kinetic, total_energy

def tau2conf(T,Wi, beta):
    A = T * Wi/(1-beta)
    A[...,0] +=1
    A[...,2] +=1
    return A

def newton_B(A00,A01,A11, tol = 1e-5):
    B01 = np.zeros_like(A00)
    def funcs(x):
        x2 = x*x
        
        sqrtA00 = np.sqrt(A00 - x2)
        sqrtA11 = np.sqrt(A11 - x2)
        if ((A00 - x2) <0).any():
            print(f'A00')
        if ((A11 - x2) <0).any():
            print(f'A11')
        f  = -A01 + x*(sqrtA00 + sqrtA11)
        df = sqrtA00 + sqrtA11 - (x2/sqrtA00 + x2/sqrtA11)
        return f, df, sqrtA00, sqrtA11

    f, df, B00, B11 = funcs(B01)
    i = 0
    while np.linalg.norm(f) > tol and i < 500:
        B01 = B01 - f/df
        f, df, B00, B11 = funcs(B01)
        i+=1
        if(np.isnan(B01).any()):
            print("The Array contain NaN values")
            break
    return B00, B01, B11

def get_data_toy(Wi, eps, dt = 0.1):
    toy = np.load(f'npz_data/expPTT_128x128_Wi-{Wi:g}_PTT-{eps:g}_VelType-stsxsy_Periodic-True_LogConf.npz', allow_pickle=True)
    solution = toy['solution']
    parameters = toy['parameters']
    L = 2*np.pi
    x = np.linspace(0,L,64)
    y = x

    X, Y = np.meshgrid(x,y)
    u = lambda x,y:   np.sin(x)*np.cos(y)
    v = lambda x,y:  -np.sin(y)*np.cos(x)

    U = u(X,Y)
    V = v(X,Y)

    vel = np.zeros((201,64,64,2))
    for i,t in enumerate(np.unique(parameters[:,:1], axis = 0)):
        vel[i,...,0] = U * np.sin(t)
        vel[i,...,1] = V * np.sin(t)
    Wis, ind = np.unique(parameters[:,1], axis = 0,return_index=True)

    j = np.where(Wis == Wi)[0]
    p = ind[j][0]
    data = solution[p:p+201]
    beta = parameters[p,2]
    data = tau2conf(data, Wi, beta)
    B00, B01, B11 = newton_B(data[...,0],data[...,1],data[...,2])
    data = np.stack((B00, B01, B11), axis=-1)

    data = np.concatenate((vel, data),axis = 3)
    q = np.moveaxis(data, [0], [2])
    TU = q[...,0].reshape((q.shape[0]**2, q.shape[2]))
    TV = q[...,1].reshape((q.shape[0]**2, q.shape[2]))
    T11 = q[...,2].reshape((q.shape[0]**2, q.shape[2]))
    T12 = q[...,3].reshape((q.shape[0]**2, q.shape[2]))
    T22 = q[...,4].reshape((q.shape[0]**2, q.shape[2]))
    T = np.concatenate((TU, TV, T11,T12,T22), axis=1).reshape(-1, q.shape[2]) # by column axis=1(intercal..), by row axis=0


    X = T[:,1:]
    Xmean = X.mean(1).reshape(-1,1)
    return X, Xmean    

def get_mesh_vtk(nome_arq):
    try:
        file = open(nome_arq, 'rt')
    except FileNotFoundError:
        print('\nNão encontrou o arquivo...\n%s\n' % nome_arq)
        return None

    # Header
    line = file.readline() # vtk datafile info
    line = file.readline() # vtk datafile info
    line = file.readline() # vtk datafile info
    line = file.readline() # vtk datafile info
    line = file.readline() # vtk datafile info

    line = file.readline() # X_COORDINATES %d float
    # Nx = int( line.split()[1] ) - 1

    line = file.readline() # Vector with x coordinates
    x = np.array(line.split()).astype(float)

    line = file.readline() # Y_COORDINATES %d float
    # Ny = int( line.split()[1] ) - 1

    line = file.readline() # Vector with y coordinates
    y = np.array(line.split()).astype(float)

    return x,y