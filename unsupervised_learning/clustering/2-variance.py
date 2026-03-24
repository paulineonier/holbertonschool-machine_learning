#!/usr/bin/env python3
"""
Module for calculating intra-cluster variance
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance

    Parameters:
    X (numpy.ndarray): shape (n, d)
    C (numpy.ndarray): shape (k, d)

    Returns:
    float: total variance, or None on failure
    """
    # Validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(C, np.ndarray) or len(C.shape) != 2 or
            X.shape[1] != C.shape[1]):
        return None

    try:
        # Calcul des distances carrées (n, k)
        distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

        # Distance minimale pour chaque point
        min_distances = np.min(distances, axis=1)

        # Somme totale
        return np.sum(min_distances)

    except Exception:
        return None
