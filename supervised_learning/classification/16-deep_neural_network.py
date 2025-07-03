#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        Constructor

        Parameters:
        - nx (int): number of input features
        - layers (list): list of number of nodes per layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            layer_input = nx if i == 0 else layers[i - 1]
            self.__weights[f'W{i+1}'] = (
                np.random.randn(layers[i], layer_input) *
                np.sqrt(2 / layer_input)
            )
            self.__weights[f'b{i+1}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Dictionary to store intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary to store weights and biases"""
        return self.__weights
