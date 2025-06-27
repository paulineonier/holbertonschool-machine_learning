#!/usr/bin/env python3
"""
Module that defines a Neuron performing binary classification.

Includes methods for forward propagation and cost computation.
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Parameters
        ----------
        nx : int
            Number of input features to the neuron.

        Raises
        ------
        TypeError
            If nx is not an integer.
        ValueError
            If nx is less than 1.
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
        """Weights vector getter."""
        return self.__W

    @property
    def b(self):
        """Bias getter."""
        return self.__b

    @property
    def A(self):
        """Activated output getter."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters
        ----------
        X : numpy.ndarray of shape (nx, m)
            Input data.

        Returns
        -------
        numpy.ndarray
            The activated output of the neuron.
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression.

        Parameters
        ----------
        Y : numpy.ndarray of shape (1, m)
            Correct labels for the input data.
        A : numpy.ndarray of shape (1, m)
            Activated output of the neuron.

        Returns
        -------
        float
            The cost (log loss).
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
