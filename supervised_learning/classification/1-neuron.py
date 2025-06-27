#!/usr/bin/env python3
import numpy as np

class Neuron:
    """Defines a single neuron performing binary classification with private attributes."""

    def __init__(self, nx):
        """
        Constructor for the Neuron class.

        Parameters:
        -----------
        nx : int
            Number of input features to the neuron.

        Raises:
        -------
        TypeError
            If nx is not an integer.
        ValueError
            If nx is less than 1.
        """
        # Validation input
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        # Private attributes
        self.__W = np.random.randn(1, nx)  # weights vector, shape (1, nx)
        self.__b = 0                       # bias, scalar
        self.__A = 0                       # activated output, scalar

    # Getter for W
    @property
    def W(self):
        """Weights vector getter."""
        return self.__W

    # Getter for b
    @property
    def b(self):
        """Bias getter."""
        return self.__b

    # Getter for A
    @property
    def A(self):
        """Activated output getter."""
        return self.__A
