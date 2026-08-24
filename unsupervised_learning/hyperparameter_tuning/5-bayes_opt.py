#!/usr/bin/env python3
"""Bayesian Optimization module with optimization loop."""
import numpy as np
from scipy.stats import norm

GaussianProcess = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Class constructor.

        Args:
            f (function): Black-box function to be optimized.
            X_init (numpy.ndarray): Inputs already sampled (shape: t, 1).
            Y_init (numpy.ndarray): Outputs for each input in X_init
                                    (shape: t, 1).
            bounds (tuple): Bounds of the space as (min, max).
            ac_samples (int): Number of acquisition sample points.
            l (float): Length parameter for the kernel.
            sigma_f (float): Standard deviation for output.
            xsi (float): Exploration-exploitation factor for acquisition.
            minimize (bool): True for minimization, False for maximization.
        """
        self.f = f
        self.gp = GaussianProcess(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], ac_samples
        ).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using EI.

        Returns:
            X_next (numpy.ndarray): Next best sample point (shape: 1,).
            EI (numpy.ndarray): Expected improvement of each potential
                                sample (shape: ac_samples,).
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            y_opt = np.min(self.gp.Y)
            improvement = y_opt - mu - self.xsi
        else:
            y_opt = np.max(self.gp.Y)
            improvement = mu - y_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(improvement)
            mask = sigma > 0
            Z[mask] = improvement[mask] / sigma[mask]

            EI = np.zeros_like(improvement)
            EI[mask] = (
                improvement[mask] * norm.cdf(Z[mask])
                + sigma[mask] * norm.pdf(Z[mask])
            )

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function using Bayesian Optimization.

        Args:
            iterations (int): Maximum number of iterations to perform.

        Returns:
            X_opt (numpy.ndarray): Optimal point (shape: 1,).
            Y_opt (numpy.ndarray): Optimal function value (shape: 1,).
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(X_next, self.gp.X)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            opt_idx = np.argmin(self.gp.Y)
        else:
            opt_idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[opt_idx]
        Y_opt = self.gp.Y[opt_idx]

        return X_opt, Y_opt