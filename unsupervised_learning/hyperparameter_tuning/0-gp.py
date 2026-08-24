#!/usr/bin/env python3
"""Gaussian Process module."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor.

        Args:
            X_init (numpy.ndarray): Inputs already sampled (shape: t, 1).
            Y_init (numpy.ndarray): Outputs for each input in X_init (shape: t, 1).
            l (float): Length-scale parameter for the kernel.
            sigma_f (float): Standard deviation for the output.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the RBF covariance kernel matrix between two matrices.

        Args:
            X1 (numpy.ndarray): Inputs of shape (m, 1).
            X2 (numpy.ndarray): Inputs of shape (n, 1).

        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n).
        """
        # Formule de la distance au carré : (x1 - x2)^2
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        
        # Formule du noyau RBF (Radial Basis Function / Noyau Gaussien)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)