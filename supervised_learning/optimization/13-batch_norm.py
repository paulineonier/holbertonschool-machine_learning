#!/usr/bin/env python3
"""
Function that normalizes an unactivated output of a neural network 
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the unactivated output of a neural network using batch normalization.

    Parameters:
    - Z: numpy.ndarray of shape (m, n), unactivated output to normalize
    - gamma: numpy.ndarray of shape (1, n), scale parameters
    - beta: numpy.ndarray of shape (1, n), shift parameters
    - epsilon: small float added to variance to avoid division by zero

    Returns:
    - Normalized and scaled Z (same shape as Z)
    """
    # Calculate mean and variance across the batch (axis=0)
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde
