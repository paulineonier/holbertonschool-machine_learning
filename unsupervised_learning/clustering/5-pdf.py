#!/usr/bin/env python3
"""
Module for calculating the PDF of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    Parameters:
    X (numpy.ndarray): shape (n, d)
    m (numpy.ndarray): shape (d,)
    S (numpy.ndarray): shape (d, d)

    Returns:
    P (numpy.ndarray): shape (n,) or None on failure
    """
    # Validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(m, np.ndarray) or len(m.shape) != 1 or
        not isinstance(S, np.ndarray) or len(S.shape) != 2 or
        X.shape[1] != m.shape[0] or
        S.shape[0] != S.shape[1] or
            S.shape[0] != m.shape[0]):
        return None

    try:
        n, d = X.shape

        # Déterminant et inverse
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)

        if det_S <= 0:
            return None

        # Normalisation
        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det_S)

        # Différence
        diff = X - m  # (n, d)

        # Exponent (sans boucle)
        exponent = -0.5 * np.sum((diff @ inv_S) * diff, axis=1)

        # PDF
        P = norm_const * np.exp(exponent)

        # Sécurité numérique
        P = np.maximum(P, 1e-300)

        return P

    except Exception:
        return None
