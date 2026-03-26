#!/usr/bin/env python3
"""
Module for determining the optimum number of clusters using variance
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Parameters:
    X (numpy.ndarray): shape (n, d)
    kmin (int): minimum number of clusters
    kmax (int): maximum number of clusters
    iterations (int): max iterations for K-means

    Returns:
    results, d_vars or None, None
    """
    # Validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2):
        return None, None

    if (not isinstance(kmin, int) or kmin <= 0):
        return None, None

    if (kmax is None):
        kmax = X.shape[0]

    if (not isinstance(kmax, int) or kmax <= 0 or kmax < kmin):
        return None, None

    if (not isinstance(iterations, int) or iterations <= 0):
        return None, None

    # Doit tester au moins 2 valeurs
    if (kmax - kmin + 1) < 2:
        return None, None

    try:
        results = []
        variances = []

        # 🔁 1 seule boucle
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            if C is None:
                return None, None

            results.append((C, clss))
            variances.append(variance(X, C))

        # Calcul des différences
        base_var = variances[0]
        d_vars = [base_var - v for v in variances]

        return results, d_vars

    except Exception:
        return None, None
