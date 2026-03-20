#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on dataset X

    Parameters:
    - X: numpy.ndarray of shape (n, d), centered data
    - var: float, fraction of variance to preserve

    Returns:
    - W: numpy.ndarray of shape (d, nd)
    """
    # Step 1: SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Step 2: Compute explained variance
    variance = S ** 2

    # Step 3: Compute cumulative variance ratio
    cumulative_variance = np.cumsum(variance) / np.sum(variance)

    # Step 4: Find number of components
    nd = np.searchsorted(cumulative_variance, var) + 1

    # Step 5: Compute W
    W = Vt.T[:, :nd]

    return W
    