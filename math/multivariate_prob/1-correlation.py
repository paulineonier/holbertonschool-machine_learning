#!/usr/bin/env python3
"""
Module that calculates a correlation matrix from
a covariance matrix.
"""

import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.

    Parameters
    ----------
    C : numpy.ndarray of shape (d, d)
        Covariance matrix

    Returns
    -------
    numpy.ndarray of shape (d, d)
        Correlation matrix
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # ----- STANDARD DEVIATIONS -----
    std = np.sqrt(np.diag(C))

    # ----- OUTER PRODUCT -----
    std_matrix = np.outer(std, std)

    # ----- CORRELATION MATRIX -----
    corr = C / std_matrix

    return corr
