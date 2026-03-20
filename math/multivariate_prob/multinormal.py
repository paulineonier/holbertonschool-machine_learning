#!/usr/bin/env python3
"""
Module that defines a Multivariate Normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    Class that represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Class constructor.

        Parameters
        ----------
        data : numpy.ndarray of shape (d, n)
            Dataset where d is the number of dimensions
            and n is the number of data points
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError(
                "data must contain multiple data points"
            )

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a given data point.

        Parameters
        ----------
        x : numpy.ndarray of shape (d, 1)
            Data point

        Returns
        -------
        float
            PDF value
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # CONSTANT TERM
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        # EXPONENT
        diff = x - self.mean
        exponent = -0.5 * (diff.T @ inv @ diff)

        # PDF
        pdf_val = norm_const * np.exp(exponent)

        return float(pdf_val)
