import numpy as np
import pandas as pd
from pathlib import Path


def PARATUCK2_imputer(X, n_latcom, reg_param, max_iter=200, eps=1e-5):
    """
    Perform matrix completion using the PARATUCK2 tensor decomposition method optimized via Alternating Least Squares.

    Parameters
    ----------
    X : np.ndarray
        Input matrix with missing values as np.nan.
    n_latcom : tuple of int
        Number of latent components for rows and columns (p, q).
    reg_param : float
        Regularization parameter for matrix inverses to improve stability.
    max_iter : int, optional
        Maximum number of ALS iterations (default is 200).
    eps : float, optional
        Convergence threshold for stopping criterion (default is 1e-5).

    Returns
    -------
    X_hat : np.ndarray
        The imputed matrix.
    A : np.ndarray
        The row factor matrix (n x p).
    R : np.ndarray
        The latent interaction matrix (p x q).
    B : np.ndarray
        The column factor matrix (m x q).
    """

    # Extract dimensions and randomly initialize factor matrices
    n, m = X.shape
    p, q = n_latcom
    A = np.random.randn(n, p) / p
    R = np.random.randn(p, q)
    B = np.random.randn(m, q) / q
    
    # Create mask Z: 1 where observed, 0 where missing
    Z = ~np.isnan(X)
    Z = Z.astype(np.float64)
    
    # Replace missing values with zero for computation compatibility
    np.nan_to_num(X, copy=False)

    # Track RMSE for convergence check
    RMSE = [
        np.linalg.norm(
            Z * (X - A @ R @ B.T), 'fro'
        ) / np.sqrt(np.sum(Z))
    ]

    for iter in range(max_iter):
        # Update A: row factor matrix
        for i, zi in enumerate(Z):
            left = R @ B.T @ np.diag(zi) @ B @ R.T + reg_param * np.eye(p)
            right = R @ B.T @ X[i].T
            A[i] = np.linalg.solve(left, right).T

        # Update R: latent interaction matrix
        M = np.zeros((p * q, p * q))
        N = np.zeros((p, q))
        for k, zk in enumerate(Z):
            # Accumulate M and N for all observed rows
            M += np.kron(B.T @ np.diag(zk) @ B, np.outer(A[k], A[k]))
            N += np.outer(A[k].T, X[k] @ B)
        R_vec = np.linalg.solve(M, N.flatten(order='F'))
        R = R_vec.reshape((p, q), order='F')

        # Update B: column factor matrix
        for j, zj in enumerate(Z.T):
            left = R.T @ A.T @ np.diag(zj) @ A @ R + reg_param * np.eye(q)
            right = R.T @ A.T @ X[:, j]
            B[j] = np.linalg.solve(left, right).T

        # Check for convergence using RMSE
        new_rmse = np.linalg.norm(
            Z * (X - A @ R @ B.T), 'fro'
        ) / np.sqrt(np.sum(Z))
        RMSE.append(new_rmse)
        if abs(RMSE[-1] - RMSE[-2]) <= eps:
            break

    # Construct the final imputed matrix from the factor matrices
    X_hat = A @ R @ B.T

    return X_hat, A, R, B


if __name__ == '__main__':

    # Load data
    data_dir = Path('./data')
    X = pd.read_csv(data_dir / 'masked_matrix.csv').to_numpy()
    X_original = pd.read_csv(data_dir / 'original_matrix.csv').to_numpy()

    # Set parameters
    n_latcom = (5, 7)  # Number of latent components
    reg_param = 200      # Regularization parameter

    # Impute missing values
    X_hat, A, R, B = PARATUCK2_imputer(X.copy(), n_latcom, reg_param)

    # Compute error on missing entries
    mask = np.logical_and(np.isnan(X), ~np.isnan(X_original))
    true_values = X_original[mask]
    imputed_values = X_hat[mask]
    median_relative_error = np.median(np.abs(true_values - imputed_values) / true_values)

    # Print results
    print(f'Imputed Matrix:\n{X_hat}')
    print(f'Median Relative Error: {median_relative_error:.4f}')