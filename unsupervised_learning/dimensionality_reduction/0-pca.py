#!/usr/bin/env python3
"""
PCA module
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters:
    X (numpy.ndarray): shape (n, d), centered dataset
    var (float): fraction of variance to preserve

    Returns:
    numpy.ndarray: weight matrix W of shape (d, nd)
    """
    n, d = X.shape

    # Compute covariance matrix
    cov = np.matmul(X.T, X) / (n - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute explained variance ratio
    explained_variance = eigenvalues / np.sum(eigenvalues)

    # Compute cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    # Find nd
    nd = np.where(cumulative_variance >= var)[0][0] + 1

    # Compute W
    W = eigenvectors[:, :nd]

    return W
