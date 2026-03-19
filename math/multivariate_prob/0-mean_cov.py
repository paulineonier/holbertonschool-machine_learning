#!/usr/bin/env python3
"""
Module that calculates the mean and covariance matrix
of a dataset.
"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a dataset.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Dataset where n is the number of data points
        and d is the number of dimensions

    Returns
    -------
    mean : numpy.ndarray of shape (1, d)
        Mean of the dataset
    cov : numpy.ndarray of shape (d, d)
        Covariance matrix of the dataset
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # MEAN
    mean = np.mean(X, axis=0, keepdims=True)

    # CENTER DATA
    X_centered = X - mean

    #  COVARIANCE
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
