import numpy as np
from .KPCA import compute_kernel_matrix

__all__ = [
    'loss_energy',
    'loss_energy_iter',
]

def loss_energy(X, X_tilde, kernel_type='oldroyd', theta=None, eps = None, norm=False):
    """
    Compute the distance metric on Reduced Kernel Hilbert Space


    X : array
        True Value
    X_tilde : array
        Approximation of X
    kernel_type: str, optional (Default: 'oldroyd')
        Kernel where the reduction is considered
    theta : float, optional
        Required parameter for kernels diferent from 'linear'
    eps : float, optional
        Non linear parameter for the non linear kernels ('ptt', 'giesekus', and 'fene-p')
    """
    K_XX  = compute_kernel_matrix(X.T, X.T, kernel_type=kernel_type, theta=theta, eps=eps)
    K_XR  = compute_kernel_matrix(X.T, X_tilde.T, kernel_type=kernel_type, theta=theta, eps=eps)
    K_RR  = compute_kernel_matrix(X_tilde.T, X_tilde.T, kernel_type=kernel_type, theta=theta, eps=eps)

    Energy = np.mean(np.diag(K_XX) - 2* np.diag(K_XR) + np.diag(K_RR))
    if norm:
        Energy /=  np.mean(np.diag(K_XX))
    return Energy

def loss_energy_iter(X, X_tilde, kernel_type='oldroyd', theta=0.1, eps= 0.4, norm = False):
    """
    Compute the distance metric on Reduced Kernel Hilbert Space (iterative version only compute the diagonal of K)
    For some reason is slower then the full matrix version


    X : array
        True Value
    X_tilde : array
        Approximation of X
    kernel_type: str, optional (Default: 'oldroyd')
        Kernel where the reduction is considered
    theta : float, optional
        Required parameter for kernels diferent from 'linear'
    eps : float, optional
        Non linear parameter for the non linear kernels ('ptt', 'giesekus', and 'fene-p')
    """
    Energy = 0
    E_sum = 0
    for i in range(X.shape[1]):
        K_XX  = compute_kernel_matrix(X[:,i:i+1].T, X[:,i:i+1].T, kernel_type=kernel_type, theta=theta, eps=eps)
        K_XR  = compute_kernel_matrix(X[:,i:i+1].T, X_tilde[:,i:i+1].T, kernel_type=kernel_type, theta=theta, eps=eps)
        K_RR  = compute_kernel_matrix(X_tilde[:,i:i+1].T, X_tilde[:,i:i+1].T, kernel_type=kernel_type, theta=theta, eps=eps)

        Energy += K_XX - 2* K_XR + K_RR
        E_sum += K_XX
    if norm:
        Energy /=  E_sum
    return Energy[0,0]/X.shape[1]