#!/usr/bin/env python3
"""
Module that defines a Neuron performing binary classification.

Includes forward propagation, cost calculation, and evaluation.
"""

import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Parameters
        ----------
        nx : int
            Number of input features.

        Raises
        ------
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Perform forward propagation.

        Parameters
        ----------
        X : numpy.ndarray of shape (nx, m)

        Returns
        -------
        numpy.ndarray
            Activated output.
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Compute cost using logistic regression.

        Parameters
        ----------
        Y : numpy.ndarray of shape (1, m)
        A : numpy.ndarray of shape (1, m)

        Returns
        -------
        float
            Cost.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate predictions.

        Parameters
        ----------
        X : numpy.ndarray of shape (nx, m)
        Y : numpy.ndarray of shape (1, m)

        Returns
        -------
        tuple
            - numpy.ndarray: predicted labels (1 or 0)
            - float: cost
        """
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, self.cost(Y, A)
