#!/usr/bin/env python3
"""
Module that defines a Neuron performing binary classification.

This class models a single neuron with private attributes for
weights, bias, and activated output, and implements forward propagation.
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
        Calculate the forward propagation of the neuron.

        Parameters
        ----------
        X : numpy.ndarray of shape (nx, m)
            Input data where nx is the number of features
            and m is the number of examples.

        Returns
        -------
        numpy.ndarray
            The activated output of the neuron (sigmoid of Z).
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
