#!/usr/bin/env python3
"""
Module for the maximization step in a GMM
"""

import numpy as np


def maximization(X, g):
    """
    Performs the M-step in the EM algorithm for a GMM

    Parameters:
    X (numpy.ndarray): shape (n, d)
    g (numpy.ndarray): shape (k, n)

    Returns:
    pi (numpy.ndarray): shape (k,)
    m (numpy.ndarray): shape (k, d)
    S (numpy.ndarray): shape (k, d, d)
    """
    # Validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(g, np.ndarray) or len(g.shape) != 2 or
            X.shape[0] != g.shape[1]):
        return None, None, None

    try:
        n, d = X.shape
        k = g.shape[0]

        # Somme des responsabilités par cluster
        Nk = np.sum(g, axis=1)  # shape (k,)

        # Priors
        pi = Nk / n  # shape (k,)

        # Means (vectorisé)
        m = (g @ X) / Nk[:, np.newaxis]  # shape (k, d)

        # Covariance matrices
        S = np.zeros((k, d, d))

        # 🔁 Une seule boucle autorisée (sur k)
        for i in range(k):
            X_centered = X - m[i]                 # shape (n, d)
            weighted = g[i][:, np.newaxis] * X_centered
            S[i] = (weighted.T @ X_centered) / Nk[i]

        return pi, m, S

    except Exception:
        return None, None, None
