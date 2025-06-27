#!/usr/bin/env python3
"""
NeuralNetwork class: 1 hidden layer for binary classification
"""

import numpy as np


class NeuralNetwork:
    """
    Neural network with one hidden layer performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Class constructor

        Args:
            nx (int): number of input features
            nodes (int): number of nodes in the hidden layer
        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
            TypeError: if nodes is not an integer
            ValueError: if nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer parameters
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Output neuron parameters
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
