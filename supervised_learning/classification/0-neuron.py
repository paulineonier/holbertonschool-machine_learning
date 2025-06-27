#!/usr/bin/env python3
"""
This module defines the Neuron class, representing a single neuron
performing binary classification.

The Neuron class includes:
- Initialization with input feature size (nx)
- Weight vector initialized with a normal distribution
- Bias initialized to zero
- Activated output initialized to zero
- Input validation on initialization parameters
"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Parameters
        ----------
        nx : int
            The number of input features to the neuron.

        Raises
        ------
        TypeError
            If nx is not an integer.
        ValueError
            If nx is less than 1.
        """
        # Step 1: Input validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Step 2: Public attributes
        # Weight vector W, initialized with a random normal distribution
        self.W = np.random.randn(1, nx)

        # Bias b, initialized to 0
        self.b = 0

        # Activated output A, initialized to 0
        self.A = 0
