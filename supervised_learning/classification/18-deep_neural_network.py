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
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Dictionary to hold intermediate values"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the deep neural network

        Parameters:
        X (numpy.ndarray): shape (nx, m), input data

        Returns:
        tuple: output of the neural network and the cache
        """
        self.__cache['A0'] = X

        for l in range(1, self.__L + 1):
            W = self.__weights[f'W{l}']
            b = self.__weights[f'b{l}']
            A_prev = self.__cache[f'A{l - 1}']
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f'A{l}'] = A

        return A, self.__cache
