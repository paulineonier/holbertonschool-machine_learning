#!/usr/bin/env python3
"""
Module for initializing a Gaussian Mixture Model
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a GMM

    Parameters:
    X (numpy.ndarray): shape (n, d)
    k (int): number of clusters

    Returns:
    pi, m, S or None, None, None
    """
    # Validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None, None, None

    try:
        n, d = X.shape

        # 🔹 Priors (uniformes)
        pi = np.full((k,), 1 / k)

        # 🔹 Means (via K-means)
        m, _ = kmeans(X, k)
        if m is None:
            return None, None, None

        # 🔹 Covariance matrices (identité)
        S = np.tile(np.eye(d), (k, 1, 1))

        return pi, m, S

    except Exception:
        return None, None, None
