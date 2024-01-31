import numpy as np

def compute_kernel_matrix(X1, X2, kernel_type, gamma = 1, eps=0.4):
    """
    Computes ther Kernel for the KPCA method using a inner product. Include the Energy Based Kernels on Hilbert Space [ref - ]
    The inputs should have shapes (n,5*m), where n is the number of snapshots and m is the number of data points (numerical mesh). Each sequence of values on
    the second axis respresents the values of (u,v,bxx,bxy,byy), respectively, of a data point, where u,v represents the velocity field and bxx,bxy,byy are 
    the elements of the simetric matrix B = SQRT(c), and C is the conformation trensor


    Parameters
    -----------
    X1, X2 :    array_like
                Inputs array, must have the same shape (n,5*m)
    kernel_type :   str ['linear', 'oldroyd', 'ptt', 'giesekus', 'fene-p']
                    inner product.
    theta   :   float
                Parameter used on all kernels, except 'linear'. Weights the contribution of the velocity and the conformation tensor
    
    eps :   float
            Extra parameters of the non-linear models

    
    Returns
    --------
    res :   ndarray
            the kernel matrix with the chooser inner product
    """
    if kernel_type == 'linear': # Standard inner product, results in PCA method
        return np.dot(X1, X2.T)

    elif kernel_type == 'oldroyd':
        # Separate the elements of the inputs

        # First input
        u1 = X1[:,0::5]
        v1 = X1[:,1::5]
        bxx1 = X1[:,2::5]
        bxy1 = X1[:,3::5]
        byy1 = X1[:,4::5]

        # Second input
        u2 = X2[:,0::5]
        v2 = X2[:,1::5]
        bxx2 = X2[:,2::5]
        bxy2 = X2[:,3::5]
        byy2 = X2[:,4::5]

        # Compute the trace of B1@B2
        tra = (bxx1@bxx2.T + 2*(bxy1@bxy2.T) + byy1@byy2.T)

        # Combine the tensor inner product with the velocity inner product
        oldroyd = u1@u2.T + v1@v2.T + gamma * tra

        return 0.5 * oldroyd # 0.5 is the scale factor that comes from the energy definition


    elif kernel_type == 'giesekus':
        # Separate the elements of the inputs

        # First input
        u1 = X1[:,0::5]
        v1 = X1[:,1::5]
        bxx1 = X1[:,2::5]
        bxy1 = X1[:,3::5]
        byy1 = X1[:,4::5]

        # Second input
        u2 = X2[:,0::5]
        v2 = X2[:,1::5]
        bxx2 = X2[:,2::5]
        bxy2 = X2[:,3::5]
        byy2 = X2[:,4::5]
        

        # Restores the first conformation tensor ( C1 = B1 @ B1)
        cxx1 = bxx1**2 + bxy1**2 - 1
        cxy1 = bxy1 * (bxx1 + byy1)
        cyy1 = byy1**2 + bxy1**2 - 1

        # Restores the second conformation tensor ( C2 = B2 @ B2)
        cxx2 = bxx2**2 + bxy2**2 - 1
        cxy2 = bxy2 * (bxx2 + byy2)
        cyy2 = byy2**2 + bxy2**2 - 1

        # Evaluate the trace of the conformation tensor
        a = cxx1 @ cxx2.T
        b = cxy1@cxy2.T
        c = cyy1@cyy2.T

        G = a + 2*b + c # matrix with the trace

        Giesekus = gamma * eps* (G) # scale with the parameters

        # Linear part of the tensor inner product (similar to the oldoyd kernel)
        oldroyd = gamma * (bxx1@bxx2.T + 2*(bxy1@bxy2.T) + byy1@byy2.T)

        # Velocity inner product (kinect energy)
        kinetic =  u1@u2.T + v1@v2.T

        # Combination of terms
        res = 0.5 * (kinetic + oldroyd + Giesekus) # 0.5 is the scale factor that comes from the energy definition
        return res
    elif kernel_type == 'ptt':
        # Separate the elements of the inputs

        # First input
        u1 = X1[:,0::5]
        v1 = X1[:,1::5]
        bxx1 = X1[:,2::5]
        bxy1 = X1[:,3::5]
        byy1 = X1[:,4::5]

        # Second input
        u2 = X2[:,0::5]
        v2 = X2[:,1::5]
        bxx2 = X2[:,2::5]
        bxy2 = X2[:,3::5]
        byy2 = X2[:,4::5]

        # Due the non linear oparation, each inner product must be evaluated separeted
        PTT = np.zeros((X1.shape[0], X2.shape[0])) #stores the results

        for i in range(X1.shape[0]): #each snapshot of X1
            for j in range(i,X2.shape[0]): #each snapshot of X2

                # trace of B1@B2
                t = (bxx1[i]*bxx2[j] + 2*(bxy1[i]*bxy2[j]) + byy1[i]*byy2[j]) - 2

                # Non linear term of the PTT model
                t = np.exp(eps*t)

                # Combine the resyults
                PTT[i,j] = t.sum()
                PTT[j,i] = PTT[i,j] # simetric result
                
        # Linear term of the PTT model (similar to the oldroyd kernel)
        tra = (bxx1@bxx2.T + 2*(bxy1@bxy2.T) + byy1@byy2.T) - 2

        # Velocity inner product (kinect energy)
        kinetic =  u1@u2.T + v1@v2.T

        # Combine terms
        res = 0.5 * (kinetic + gamma * (PTT * tra + 2))  # 0.5 is the scale factor that comes from the energy definition
        return res
    
    elif kernel_type == 'fene-p':
        dx = 1
        u1 = X1[:,0::5]
        v1 = X1[:,1::5]
        bxx1 = X1[:,2::5]
        bxy1 = X1[:,3::5]
        byy1 = X1[:,4::5]

        u2 = X2[:,0::5]
        v2 = X2[:,1::5]
        bxx2 = X2[:,2::5]
        bxy2 = X2[:,3::5]
        byy2 = X2[:,4::5]

        
        fenep = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(i,X2.shape[0]):
                t = (bxx1[i]*bxx2[j] + 2*(bxy1[i]*bxy2[j]) + byy1[i]*byy2[j])
                c = (eps*t)/(eps - t) 
                fenep[i,j] = c.sum()
                fenep[j,i] = fenep[i,j]
                

        # print(coef.min(), coef.max())
        ptt = u1@u2.T + v1@v2.T + gamma * (fenep)
        return 0.5 * dx * dx * ptt

    else:
        raise ValueError("Invalid kernel type. Options are 'linear', 'oldroyd', 'giesekus', 'ptt' or 'fene-p'.")


def kpca(X, n_components=2, kernel='linear', gamma=None, norm='MULT', eps = None):
    """
    Perform Kernel Principal Component Analysis (PCA) on input data X.

    Parameters
    ----------
    X : Data matrix
        Input data.
    n_components : int, optional (default=2)
        Number of components to keep.
    kernel : str, optional (default='linear')
        Kernel function to use. Options are 'linear', 'oldroyd', 'giesekus', 'ptt' or 'fene-p'.
    gamma : float, optional (default=None)
        Gamma parameter for 'oldroyd', 'giesekus', 'ptt' and 'fene-p' kernels. Not needed for other kernels.
    norm : str, optional (default='MULT')
        normalization to use. Options are 'MULT', 'DIV' or 'NONE'.
    eps : float, optional (default=None)
        Gamma parameter for 'giesekus', 'ptt' and 'fene-p' kernels. Not needed for other kernels.

    Returns
    -------
    X_kpca : array, shape (n_samples, n_components)
        Transformed data in reduced-dimensional space.
    """

    

    # Compute the kernel matrix
    K = compute_kernel_matrix(X, X, kernel, gamma, eps)

    # Center the kernel matrix
    K_centered = K - np.mean(K, axis=0)
    K_centered = K_centered - np.mean(K_centered, axis=1).reshape(-1, 1)

    
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
    if norm == 'MULT': # Multiply by the eigenvector
        eigenvectors_normalized = eigenvectors * np.sqrt(eigenvalues[:n_components])
    elif norm == 'DIV': # Divide by the eigenvector
        eigenvectors_normalized = eigenvectors / np.sqrt(eigenvalues[:n_components])
    elif norm == 'NONE': # No Normalization
        eigenvectors_normalized = eigenvectors
    else:
        raise ValueError(f"Invalid norm type. Options are 'MULT', 'DIV' or 'None' (given: '{norm}')")
    
    # Project input data onto the eigenvectors
    X_kpca = np.dot(K_centered, eigenvectors_normalized)

    return X_kpca, eigenvectors_normalized, eigenvalues, K_centered

