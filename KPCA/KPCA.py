import numpy as np

__all__ = [
    'compute_kernel_matrix',
    'kpca'
    ]

def compute_kernel_matrix(X1, X2, kernel_type, theta, eps=None, dx = 1, dy = None):
    if kernel_type == 'linear':
        return np.dot(X1, X2.T)

    else:
        #Separate the variables
        u1 = X1[:,0::5] # first velocity component
        v1 = X1[:,1::5] # second velocity component
        bxx1 = X1[:,2::5] # component of the square root of the conformation tensor
        bxy1 = X1[:,3::5] # component of the square root of the conformation tensor
        byy1 = X1[:,4::5] # component of the square root of the conformation tensor

        u2 = X2[:,0::5] # first velocity component
        v2 = X2[:,1::5] # second velocity component
        bxx2 = X2[:,2::5] # component of the square root of the conformation tensor
        bxy2 = X2[:,3::5] # component of the square root of the conformation tensor
        byy2 = X2[:,4::5] # component of the square root of the conformation tensor

        if dy is None:
            dy = dx
        area = dx*dy
        if kernel_type == 'oldroyd':
            tra = ((area*bxx1)@bxx2.T + 2*((area*bxy1)@bxy2.T) + (area*byy1)@byy2.T) # Trace of (sqrt(C1) * sqrt(C2))
            oldroyd = (area*u1)@u2.T + (area*v1)@v2.T + theta * tra
            return .5 * oldroyd

        elif kernel_type == 'giesekus':
            # Reconstruction of the conformation tensor minus 1
            cxx1 = bxx1**2 + bxy1**2 - 1
            cxy1 = bxy1 * (bxx1 + byy1)
            cyy1 = byy1**2 + bxy1**2 - 1

            # Reconstruction of the conformation tensor minus 1
            cxx2 = bxx2**2 + bxy2**2 - 1
            cxy2 = bxy2 * (bxx2 + byy2)
            cyy2 = byy2**2 + bxy2**2 - 1

            #Operatations for computation of C1 @ C2
            a = (area * cxx1) @ cxx2.T
            b = (area * cxy1) @ cxy2.T
            c = (area * cyy1) @ cyy2.T

            #Trace of C1 @ C2
            G = a + 2*b + c

            #Multiply weights
            Giesekus = theta * eps * G

            # Kinectic part
            kinetic =  (area * u1)@u2.T + (area * v1)@v2.T

            # Same term as oldroyd kernel
            oldroyd = theta * ((area * bxx1)@bxx2.T + 2*((area * bxy1)@bxy2.T) + (area * byy1)@byy2.T)

            #Add everything
            res = (kinetic + oldroyd + Giesekus)

            return res *.5
        elif kernel_type == 'ptt':
            PTT = np.zeros((X1.shape[0], X2.shape[0])) # store the computations

            for i in range(X1.shape[0]): # For each snapshot of X1
                for j in range(X2.shape[0]): # For each snapshot of X2

                    t = (bxx1[i]*bxx2[j] + 2*(bxy1[i]*bxy2[j]) + byy1[i]*byy2[j]) - 2 # Trace of (sqrt(C1) * sqrt(C2) - I) 
                    t = area * (np.exp(eps*t) * t + 2) # PTT term
                    PTT[i,j] = t.sum() # "integrate"

            ptt = (area * u1)@u2.T + (area * v1)@v2.T + theta * PTT # Add the kinectic term
            return 0.5 * ptt

        elif kernel_type == 'fene-p':

            fenep = np.zeros((X1.shape[0], X2.shape[0]))# store the computations
            for i in range(X1.shape[0]): # For each snapshot of X1
                for j in range(X2.shape[0]): # For each snapshot of X2
                    t = (bxx1[i]*bxx2[j] + 2*(bxy1[i]*bxy2[j]) + byy1[i]*byy2[j]) # Trace of (sqrt(C1) * sqrt(C2)) 
                    c = area * (eps*t)/(eps - t) 
                    fenep[i,j] = c.sum()

            fene = (area * u1)@u2.T + (area * v1)@v2.T + theta * (fenep) # Add the kinectic term
            return 0.5 * fene

        
        else:
            raise ValueError("Invalid kernel type. Options are 'linear', 'oldroyd', 'ptt', 'giesekus' or 'fene-p'.")
        
def kpca(X, n_components=2, kernel='linear', gamma=None, eps = None, norm='DIV', dx = 1, dy = None):
    """
    Perform Kernel Principal Component Analysis (PCA) on input data X.

    Parameters
    ----------
    X : Data matrix
        Input data.
    n_components : int, optional (default=2)
        Number of components to keep.
    kernel : str, optional (default='linear')
        Kernel function to use. Options are 'linear', 'oldroyd', 'ptt', 'giesekus' or 'fene-p'.
    gamma : float, optional (default=None)
        Gamma parameter for 'oldroyd', 'ptt', 'giesekus' or 'fene-p'." kernels. Not needed for other kernels.
    eps : float, optional (default=None)
        non-linear parameter for 'ptt', 'giesekus' or 'fene-p'." kernels. Not needed for other kernels.
    eps : str, optional (default='DIV')
        Normalize the eigenvectors by the eigenvalues
    dx, dy : float, optinal 
        for energy based kernels integration

    Returns
    -------
    X_kpca : array, shape (n_samples, n_components)
        Transformed data in reduced-dimensional space.
    eigenvectors_normalized : array, shape (n_samples, n_components)
        Reduced eigenvector normalized.
    eigenvalues : array, shape (n_samples, n_components)
        Reduced eigenvector.
    K_centered : array, shape (n_samples, n_components)
        Kernel matrix
    """


    # Compute the kernel matrix
    K = compute_kernel_matrix(X, X, kernel, gamma, eps,dx,dy)
    #traceK = np.trace(K)
    #print(traceK)
    # Center the kernel matrix
    K_centered = K - np.mean(K, axis=0)
    K_centered = K_centered - np.mean(K_centered, axis=1).reshape(-1, 1)
    # K_centered = K - np.mean(K, axis=1).reshape(-1, 1)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    # Sort eigenvalues and eigenvectors in descending order
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Select top n_components eigenvectors
    eigenvectors = eigenvectors[:, :n_components]

    # Normalize eigenvectors
    norm = norm.upper()
    if norm == 'MULT':
        eigenvectors_normalized = eigenvectors * np.sqrt(eigenvalues[:n_components])
    elif norm == 'DIV':
        eigenvectors_normalized = eigenvectors / np.sqrt(eigenvalues[:n_components])
    elif norm == 'NONE':
        eigenvectors_normalized = eigenvectors
    else:
        raise ValueError(f"Invalid norm type. Options are 'MULT', 'DIV' or 'None' (given: '{norm}')")
    
    # Project input data onto the eigenvectors
    X_kpca = np.dot(K_centered, eigenvectors_normalized)

    return X_kpca, eigenvectors_normalized, eigenvalues, K_centered