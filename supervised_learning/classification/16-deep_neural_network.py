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

        Args:
            nx (int): number of input features
            layers (list): list representing the number of nodes in each layer

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
            TypeError: if layers is not a list of positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            layer_size = layers[l - 1]
            prev_size = nx if l == 1 else layers[l - 2]
            self.__weights[f'W{l}'] = (np.random.randn(layer_size, prev_size)
                                       * np.sqrt(2 / prev_size))
            self.__weights[f'b{l}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for L: number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache: dictionary to store forward propagation values"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights: dictionary to store weights and biases"""
        return self.__weights
