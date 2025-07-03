#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor

        Parameters:
        nx (int): number of input features
        layers (list): number of nodes in each layer of the network

        Raises:
        TypeError: if nx is not an integer
        ValueError: if nx is less than 1
        TypeError: if layers is not a list of positive integers
        """
        # Validations
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)             # Number of layers
        self.cache = {}                  # Dictionary to store intermediate values
        self.weights = {}                # Dictionary to store weights and biases

        # He et al. initialization
        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_layer_size = nx if l == 1 else layers[l - 2]

            self.weights[f'W{l}'] = (np.random.randn(layer_size, prev_layer_size)
                                     * np.sqrt(2 / prev_layer_size))
            self.weights[f'b{l}'] = np.zeros((layer_size, 1))
