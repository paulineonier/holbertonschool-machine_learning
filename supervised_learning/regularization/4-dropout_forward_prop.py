#!/usr/bin/env python3
"""
Module 4-dropout_forward_prop
This module defines a function to perform forward propagation with
Dropout regularization in a deep neural network.

Functions:
    dropout_forward_prop(X, weights, L, keep_prob):
        Conducts forward propagation with Dropout.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X (numpy.ndarray): Shape (nx, m), input data for the network.
            - nx: number of input features
            - m: number of data points
        weights (dict): Dictionary of the weights and biases of the
            neural network.
            - keys: "Wl", "bl" where l is the layer number.
        L (int): Number of layers in the network.
        keep_prob (float): Probability that a node will be kept
            (between 0 and 1).

    Returns:
        dict: Dictionary containing the outputs of each layer
        and the dropout mask used on each hidden layer.
        Format:
            - "Al": activations at layer l
            - "Dl": dropout mask for layer l (if l < L)
    """
    cache = {}
    cache['A0'] = X
    m = X.shape[1]

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]

        Z = np.matmul(W, A_prev) + b

        if layer == L:
            # Softmax activation
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh activation
            A = np.tanh(Z)

            # Dropout mask
            D = np.random.rand(*A.shape) < keep_prob
            cache['D' + str(layer)] = D.astype(int)

            # Apply mask and scale
            A = (A * cache['D' + str(layer)]) / keep_prob

        cache['A' + str(layer)] = A

    return cache
