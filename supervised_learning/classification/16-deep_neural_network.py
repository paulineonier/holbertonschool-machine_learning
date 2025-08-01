#!/usr/bin/env python3
"""
Module that defines the DeepNeuralNetwork class
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor

        Parameters
        ----------
        nx : int
            Number of input features
        layers : list
            List representing the number of nodes in each layer

        Raises
        ------
        TypeError
            If nx is not an int
            If layers is not a list of positive integers
        ValueError
            If nx is less than 1
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
            self.__weights[f"W{i+1}"] = (
                np.random.randn(layers[i], layer_input) * np.sqrt(2 / layer_input)
            )
            self.__weights[f"b{i+1}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        Number of layers in the neural network

        Returns
        -------
        int
        """
        return self.__L

    @property
    def cache(self):
        """
        Dictionary holding all intermediary values of the network

        Returns
        -------
        dict
        """
        return self.__cache

    @property
    def weights(self):
        """
        Dictionary holding all weights and biases of the network

        Returns
        -------
        dict
        """
        return self.__weights
