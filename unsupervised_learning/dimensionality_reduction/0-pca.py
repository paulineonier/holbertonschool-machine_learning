#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on dataset X
    """

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Explained variance ratio
    explained_variance = (S ** 2) / np.sum(S ** 2)

    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    # Number of components
    nd = np.where(cumulative_variance >= var)[0][0] + 1

    # Projection matrix
    W = Vt.T[:, :nd]

    return W
