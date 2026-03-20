#!/usr/bin/env python3
"""
PCA transformation module
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset and reduces its dimensionality.

    Parameters:
    X (numpy.ndarray): shape (n, d)
    ndim (int): new dimensionality

    Returns:
    numpy.ndarray: transformed dataset of shape (n, ndim)
    """
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Step 3: Select top ndim components
    W = Vt.T[:, :ndim]

    # Step 4: Project data
    T = np.matmul(X_centered, W)

    return T
