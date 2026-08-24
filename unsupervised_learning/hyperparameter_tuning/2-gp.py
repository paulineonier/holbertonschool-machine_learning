#!/usr/bin/env python3
"""Gaussian Process module with prediction and update methods."""
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
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        )
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """Predicts the mean and variance of points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): Points whose mean and variance should be
                                 calculated (shape: s, 1).

        Returns:
            mu (numpy.ndarray): Mean for each point in X_s (shape: s,).
            sigma (numpy.ndarray): Variance for each point in X_s (shape: s,).
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = np.dot(K_s.T, np.dot(K_inv, self.Y)).reshape(-1)
        sigma = np.diag(K_ss - np.dot(K_s.T, np.dot(K_inv, K_s)))

        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates a Gaussian Process with a new sample point.

        Args:
            X_new (numpy.ndarray): New sample point (shape: 1,).
            Y_new (numpy.ndarray): New sample function value (shape: 1,).
        """
        # Redimensionnement de X_new et Y_new en forme (1, 1) pour np.vstack
        X_new_reshaped = X_new.reshape(-1, 1)
        Y_new_reshaped = Y_new.reshape(-1, 1)

        # Empilement vertical pour mettre à jour self.X et self.Y
        self.X = np.vstack((self.X, X_new_reshaped))
        self.Y = np.vstack((self.Y, Y_new_reshaped))

        # Recalcul complet de la matrice de covariance K avec le nouveau jeu de données
        self.K = self.kernel(self.X, self.X)