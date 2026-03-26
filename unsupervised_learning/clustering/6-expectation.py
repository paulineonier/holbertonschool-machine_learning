#!/usr/bin/env python3
"""
Module for the expectation step in a GMM
"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the E-step in the EM algorithm for a GMM

    Parameters:
    X (numpy.ndarray): shape (n, d)
    pi (numpy.ndarray): shape (k,)
    m (numpy.ndarray): shape (k, d)
    S (numpy.ndarray): shape (k, d, d)

    Returns:
    g (numpy.ndarray): shape (k, n) of posterior probabilities
    L (float): total log likelihood
    """
    # Validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(pi, np.ndarray) or len(pi.shape) != 1 or
        not isinstance(m, np.ndarray) or len(m.shape) != 2 or
        not isinstance(S, np.ndarray) or len(S.shape) != 3 or
        X.shape[1] != m.shape[1] or
        pi.shape[0] != m.shape[0] or
        S.shape[0] != m.shape[0] or
        S.shape[1] != S.shape[2] or
            S.shape[1] != X.shape[1]):
        return None, None

    try:
        n, d = X.shape
        k = pi.shape[0]

        # Posterior matrix g (k, n)
        g = np.zeros((k, n))

        # 🔁 On boucle sur k clusters (max 1 boucle autorisée)
        for i in range(k):
            g[i, :] = pi[i] * pdf(X, m[i], S[i])

        # Normalisation pour que chaque colonne somme à 1
        g_sum = np.sum(g, axis=0)
        g /= g_sum

        # Log-likelihood totale
        L = np.sum(np.log(g_sum))

        return g, L

    except Exception:
        return None, None
