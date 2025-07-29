#!/usr/bin/env python3
"""Module that calculates normalization constants using numpy"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of a dataset.

    Args:
        X (np.ndarray): matrix of shape (m, nx)
            m: number of data points
            nx: number of features

    Returns:
        tuple: (mean, std) both numpy arrays of shape (nx,)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
