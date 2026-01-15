import numpy as np
from sklearn.gaussian_process.kernels import Matern
"""
This module provides a set of functions and classes for performing Kernel Principal Component Analysis (KPCA) 
and working with various kernel functions. It includes implementations of custom kernel functions, 
kernel matrix computation, and KPCA transformations.
Modules:
--------
- `compute_kernel_matrix`: Computes the kernel matrix for a given kernel type.
- `kpca`: Performs Kernel Principal Component Analysis (KPCA) on input data.
- `apply_kpca`: Applies a precomputed KPCA transformation to new data.
- `KernelPCA`: A class for performing KPCA with additional functionality for transformation, 
    inverse transformation, and model saving/loading.
- `chol`: Performs Cholesky decomposition with a probabilistic selection of rows.
Kernel Types:
-------------
The module supports the following kernel types:
- `linear`: Linear kernel.
- `poly`: Polynomial kernel.
- `rbf`: Radial Basis Function (Gaussian) kernel.
- `sigmoid`: Sigmoid kernel.
- `cosine`: Cosine similarity kernel.
- `matern`: Matern kernel (requires `scikit-learn`).
- `oldroyd`: Custom kernel for Oldroyd-B model.
- `giesekus`: Custom kernel for Giesekus model.
- `ptt`: Custom kernel for Phan-Thien-Tanner (PTT) model.
- `fene-p`: Custom kernel for FENE-P model.
Classes:
--------
- `KernelPCA`: A class for performing Kernel PCA with additional methods for transformation, 
    inverse transformation, and model persistence.
Functions:
- `chol(A, k)`: Performs Cholesky decomposition with probabilistic row selection.
- `compute_kernel_matrix(X1, X2, kernel_type, theta=None, eps=None, dx=1, dy=None)`: 
    Computes the kernel matrix for the specified kernel type.
- `kpca(X, n_components=2, kernel='linear', theta=None, eps=None, norm='DIV', dx=1, dy=None)`: 
    Performs Kernel PCA on the input data.
- `apply_kpca(X, X_train, Q, K_fit, kernel='linear', theta=None, eps=None, dx=1, dy=None)`: 
    Applies a precomputed KPCA transformation to new data.
Usage:
------
This module is designed for advanced users who need to perform Kernel PCA with custom kernel functions 
or work with energy-based kernels for specific applications. It provides flexibility in defining 
and using custom kernels for specialized tasks.
Dependencies:
-------------
- `numpy`: For numerical computations.
- `scikit-learn`: For Matern kernel (optional, required only for `matern` kernel type).
"""

__all__ = [
    'compute_kernel_matrix',
    'kpca',
    'apply_kpca',
    'KernelPCA',
    'chol'
    ]



def chol(A,k):
    """
    Performs a probabilistic Cholesky decomposition of a given matrix.
    This function computes a low-rank approximation of the input matrix `A` 
    using a probabilistic Cholesky decomposition approach. It selects `k` 
    columns iteratively based on a probability distribution derived from 
    the diagonal elements of the matrix.
    Parameters:
    -----------
    A : numpy.ndarray
        A symmetric positive semi-definite matrix of shape (n, n) to be decomposed.
    k : int
        The rank of the decomposition, i.e., the number of columns to select.
    Returns:
    --------
    F : numpy.ndarray
        A matrix of shape (n, k) containing the low-rank approximation factors.
    s : list
        A list of indices corresponding to the selected columns during the decomposition.
    Notes:
    ------
    - The input matrix `A` must be symmetric and positive semi-definite.
    - The function uses a probabilistic approach to select columns based on 
        the diagonal elements of the matrix.
    - The diagonal elements are updated iteratively to reflect the residual 
        variance after each column selection.
    Example:
    --------
    >>> import numpy as np
    >>> A = np.array([[4, 2], [2, 3]])
    >>> k = 1
    >>> F, s = chol(A, k)
    >>> print(F)
    >>> print(s)
    """


    n = A.shape[0]
    F = np.zeros((n,k))
    d = np.diag(A)
    s = list()
    indexes = np.arange(n)
    for i in range(k):
        si = np.random.choice(indexes,1, p=d/d.sum()).item()
        g = A[:,si]
        g = g - F[:, :i] @ F[si,:i].T
        # print(g)
        F[:,i] = (g /np.sqrt(g[si]))
        d = d - np.abs(F[:,i])**2
        d[d<0] = 0
        s.append(si)
    return F,s

def compute_kernel_matrix(X1, X2, kernel_type, theta=None, eps=None, dx = 1, dy = None):
    if kernel_type == 'linear':
        return np.dot(X1, X2.T)
    elif kernel_type == 'poly':
        return (np.dot(X1, X2.T) + 1) ** eps
    elif kernel_type == 'rbf':
        eps = 1.0 / (2 * eps**2)
        X1_norm = np.sum(X1**2, axis=1)
        X2_norm = np.sum(X2**2, axis=1)
        dist = np.outer(X1_norm, np.ones(X2.shape[0])) + np.outer(np.ones(X1.shape[0]), X2_norm) - 2 * np.dot(X1, X2.T)
        return np.exp(-eps * dist)
    elif kernel_type == 'sigmoid':
        return np.tanh(eps * np.dot(X1, X2.T) + 1)
    elif kernel_type == 'cosine':
        return np.dot(X1, X2.T) / (np.linalg.norm(X1, axis=1, keepdims=True) * np.linalg.norm(X2, axis=1, keepdims=True).T)
    elif kernel_type == 'matern':
        gamma, nu = eps
        return Matern(length_scale=gamma, nu=nu)(X1,X2)

    else:

        #Separate the variables
        u1 = X1[...,0::5] # first velocity component
        v1 = X1[...,1::5] # second velocity component
        bxx1 = X1[...,2::5] # component of the square root of the conformation tensor
        bxy1 = X1[...,3::5] # component of the square root of the conformation tensor
        byy1 = X1[...,4::5] # component of the square root of the conformation tensor

        u2 = X2[...,0::5] # first velocity component
        v2 = X2[...,1::5] # second velocity component
        bxx2 = X2[...,2::5] # component of the square root of the conformation tensor
        bxy2 = X2[...,3::5] # component of the square root of the conformation tensor
        byy2 = X2[...,4::5] # component of the square root of the conformation tensor

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
            raise ValueError("Invalid kernel type. Options are 'linear', 'cosine', 'sigmoid', 'poly', 'rbf', 'oldroyd', 'ptt', 'giesekus' or 'fene-p'.")
        
def kpca(X, n_components=2, kernel='linear', theta=None, eps = None, norm='DIV', dx = 1, dy = None):
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
    theta : float, optional (default=None)
        theta parameter for 'oldroyd', 'ptt', 'giesekus' or 'fene-p'." kernels. Not needed for other kernels.
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
    K = compute_kernel_matrix(X, X, kernel, theta, eps,dx,dy)
    #traceK = np.trace(K)
    #print(traceK)
    # Center the kernel matrix
    
    K_row = np.mean(K, axis=0)
    K_col = np.mean(K, axis=1)[:,None]
    K_all = np.mean(K)
    K_centered = K - K_row - K_col + K_all
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
    X_kpca = eigenvectors_normalized * eigenvalues[:n_components]

    return X_kpca, eigenvectors_normalized, eigenvalues, K_centered, K

def apply_kpca(X, X_train, Q, K_fit,kernel='linear', theta=None, eps = None, dx = 1, dy = None):
    m = K_fit.shape[0]
    ones = np.full((1,m), 1/m)
    M = np.eye(m) - np.full((m,m), 1/m)
    K = compute_kernel_matrix(X, X_train, kernel, theta, eps,dx,dy)
    # K_row = np.mean(K, axis=0)
    # K_col = np.mean(K, axis=1)[:,None]
    # K_all = np.mean(K)
    # K_centered = K - K_row - K_col + K_all
    X_kpca = (K - ones@K_fit) @ M@Q

    return X_kpca

class KernelPCA():
    """
        Kernel Principal Component Analysis (Kernel PCA) implementation.
        This class provides methods to perform Kernel PCA, a dimensionality reduction technique
        that uses kernel methods to project data into a lower-dimensional space.
        Attributes
        _is_fitted : bool
            Indicates whether the model has been fitted.
        K_fit : array
            Kernel matrix computed during fitting.
        X_fit : array
            Input data used during fitting.
        kernel : str
            Kernel function used during fitting.
        thetas_fit : array
            Theta parameters used during fitting.
        eps_fit : array
            Epsilon parameters used during fitting.
        center : bool
            Indicates whether the kernel matrix is centered.
        degree : int
            Degree of polynomial expansion used in the reconstruction.
        n_components : int
        all_eignevectors : array
            All eigenvectors computed during fitting.
        normalized_eigenvector : array
            Normalized eigenvectors corresponding to the top components.
        eigenvalues : array
            Eigenvalues corresponding to the eigenvectors.
        R : array
            Reconstruction matrix.
        U_fit : array
            Matrix used for transformation.
            Initialize the KernelPCA object.
            Sets up the initial state of the object, including setting the `_is_fitted` attribute to False.
    """
    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self,X, n_components=2, kernel='linear', theta=None, eps = None, dx = 1, dy = None, degree = 1, center = True , use_chol = 0):
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
        theta : float, optional (default=None)
            theta parameter for 'oldroyd', 'ptt', 'giesekus' or 'fene-p'." kernels. Not needed for other kernels.
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
        K = compute_kernel_matrix(X, X, kernel, theta, eps,dx,dy)

        # store information

        self.K_fit = K
        self.X_fit = X
        self.kernel = kernel

        # for inverse transform:
        #Get reconst matrix
        try:
            self.thetas_fit = np.diag(theta)[None,:]
        except ValueError as e:
            self.thetas_fit = theta
        try:
            self.eps_fit = np.diag(eps)[None,:]
        except ValueError as e:
            self.eps_fit = eps
        self.center = center
        self.train_R(degree, center,n_components, use_chol,recalc_eig=True)
        self._is_fitted = True


    def train_R(self,degree = 1, center= True, n_components = 3, use_chol = 0, recalc_eig = False):
            """
            Train the R matrix using kernel PCA and regression.
            Parameters:
            -----------
            degree : int, optional (default=1)
                The degree of polynomial features to be used in the regression.
            center : bool, optional (default=True)
                Whether to center the kernel matrix. If the centering option changes, 
                eigen decomposition will be recalculated.
            n_components : int, optional (default=3)
                The number of principal components to retain during kernel PCA.
            use_chol : int, optional (default=0)
                If greater than 0, Cholesky decomposition is used for kernel matrix 
                decomposition. The value determines the rank for approximation.
            recalc_eig : bool, optional (default=False)
                If True, forces recalculation of eigen decomposition even if the kernel 
                matrix has not changed.
            Attributes:
            -----------
            self.all_eignevectors : ndarray
                All eigenvectors of the kernel matrix.
            self.normalized_eigenvector : ndarray
                Normalized eigenvectors corresponding to the top `n_components` eigenvalues.
            self.eigenvalues : ndarray
                Eigenvalues of the kernel matrix.
            self.R : ndarray
                The regression matrix computed using least squares.
            Returns:
            --------
            None
                The method updates the instance attributes with the computed values.
            Notes:
            ------
            - The method performs kernel PCA on the kernel matrix `self.K_fit` and 
                computes the regression matrix `R` using least squares.
            - If `use_chol > 0`, Cholesky decomposition is used for kernel matrix 
                decomposition; otherwise, eigen decomposition is used.
            - Polynomial features are generated based on the degree specified, and 
                additional features are added depending on the kernel type.
            """
        
            # Eigen decomposition 
            self.degree = degree
            if center != self.center:
                self.center = center
                recalc_eig = True
            self.n_components = n_components
            K = self.K_fit
            if not self._is_fitted or recalc_eig:
                if center:
                    K_row = np.mean(K, axis=0)
                    K_col = np.mean(K, axis=1)[:,None]
                    K_all = np.mean(K)
                    K_centered = K - K_row - K_col + K_all
                else:
                    K_centered = K

                if use_chol > 0:
                    F,_ = chol(K_centered,use_chol) # how to get a good value for k?
                    eigenvectors,singularvalues,_ = np.linalg.svd(F,full_matrices=False)
                    self.all_eignevectors = eigenvectors
                    self.normalized_eigenvector = eigenvectors[:, :self.n_components] / singularvalues[:self.n_components]
                    self.eigenvalues = singularvalues**2
                else:
                    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
                    # Sort eigenvalues and eigenvectors in descending order
                    indices = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[indices]
                    eigenvectors = eigenvectors[:, indices]

                    # Select top n_components eigenvectors
                    self.all_eignevectors = eigenvectors
                    eigenvectors = eigenvectors[:, :self.n_components]
                    self.normalized_eigenvector = eigenvectors / np.sqrt(eigenvalues[:self.n_components])
                    self.eigenvalues = eigenvalues
            else:
                self.normalized_eigenvector = self.all_eignevectors[:, :self.n_components] / np.sqrt(self.eigenvalues[:self.n_components])
                        # for transform:
            # m = self.K_fit.shape[0]
            # M = np.eye(m) - np.full((m,m), 1.0/m)
            # self.U_fit = M@self.normalized_eigenvector

            X_kpca = self.normalized_eigenvector * self.eigenvalues[:self.n_components]
            sqrt_theta = np.sqrt(self.thetas_fit).T 
            ones_fit = np.ones((self.X_fit.shape[0],1))
            Q2 = np.concatenate([ones_fit]+ [ones_fit / sqrt_theta ]  + [(X_kpca)**(k+1) for k in range(self.degree)] + [(X_kpca / sqrt_theta)**(k+1) for k in range(self.degree)], axis=1)
            if self.kernel == 'giesekus':
                sqrt_alpha = np.sqrt(self.eps_fit).T * sqrt_theta
                Q2 = np.concatenate([Q2] + [ones_fit / sqrt_alpha] + [(X_kpca / sqrt_alpha)**(k+1) for k in range(self.degree)], axis=1)
            R, _, _, _ = np.linalg.lstsq(Q2, self.X_fit, rcond=None)
            self.R = R.T

    def transform(self,X, theta=None, eps = None, dx = 1, dy = None):
        if theta is not None:
            try:
                theta = np.sqrt(theta@self.thetas_fit)
            except TypeError:
                theta = np.sqrt(theta*self.thetas_fit)
        if self.kernel == 'giesekus':
            try:
                eps = np.sqrt(eps@self.eps_fit)
            except TypeError:
                eps = np.sqrt(eps*self.eps_fit)
        K = compute_kernel_matrix(X, self.X_fit, self.kernel, theta, eps,dx,dy)
        # ones  = np.full(K.shape, 1/self.K_fit.shape[0])
        # X_kpca = (K - ones@self.K_fit) @ self.U_fit
        if self.center:
            K_row = np.mean(self.K_fit, axis=0)
            K_col = np.mean(self.K_fit, axis=1)[:,None]
            K_all = np.mean(self.K_fit)
            K_centered = K - K_row - K_col + K_all
            X_kpca = K_centered @ self.normalized_eigenvector
        else:
            X_kpca = K @ self.normalized_eigenvector
        return X_kpca
    
    def invert_transform(self, Phi, theta, eps = None):
        sqrt_theta = np.sqrt(theta)
        ones_phi = np.ones((Phi.shape[0],1))
        Phi_ext = np.concatenate([ones_phi] + [ones_phi / sqrt_theta] + [Phi**(k+1) for k in range(self.degree)]+ [(Phi / sqrt_theta)**(k+1) for k in range(self.degree)], axis=1)
        if self.kernel == 'giesekus':
            sqrt_alpha = np.sqrt(eps) * sqrt_theta
            Phi_ext = np.concatenate([Phi_ext] + [ones_phi / sqrt_alpha] + [(Phi / sqrt_alpha)**(k+1) for k in range(self.degree)], axis=1)
        #reconstruction
        X = self.R@Phi_ext.T
        return X
    
    def save_model(self, filename, compressed = True):
        if self._is_fitted:
            if compressed:
                np.savez_compressed(filename, eigvec = self.normalized_eigenvector, eigvalue = self.eigenvalues, K = self.K_fit, X = self.X_fit, kernel = self.kernel, U = self.U_fit, R = self.R)
            else:
                np.savez(filename, eigvec = self.normalized_eigenvector, eigvalue = self.eigenvalues, K = self.K_fit, X = self.X_fit, kernel = self.kernel, U = self.U_fit, R = self.R)
        else:
            print('Nothing to save!')


    def load_model(self, filename):
        data = np.load(filename, allow_pickle=True)
        self._is_fitted = True
        self.normalized_eigenvector = data['eigvec']
        self.eigenvalues = data['eigvalue']
        self.K_fit = data['K']
        self.X_fit = data['X']
        self.kernel = data['kernel']
        self.U_fit = data['U']
        self.R = data['R']