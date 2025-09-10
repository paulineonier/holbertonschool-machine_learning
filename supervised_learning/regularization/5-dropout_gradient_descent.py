#!/usr/bin/env python3
"""
Module 5-dropout_gradient_descent
This module defines a function to update weights of a neural network
using gradient descent with Dropout regularization.

Functions:
    dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
        Updates the weights in-place.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y (numpy.ndarray): Shape (classes, m), one-hot labels.
            - classes: number of classes
            - m: number of examples
        weights (dict): Dictionary of network weights and biases:
            - "Wl", "bl"
        cache (dict): Dictionary containing activations "Al"
            and dropout masks "Dl" from forward propagation.
        alpha (float): Learning rate.
        keep_prob (float): Probability that a node is kept.
        L (int): Number of layers.

    Returns:
        None. The weights are updated in-place.
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y  # dérivée softmax

    for layer in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]

        # Gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update
        weights["W" + str(layer)] = W - alpha * dW
        weights["b" + str(layer)] = weights["b" + str(layer)] - alpha * db

        if layer > 1:
            # Backprop dZ for hidden layers (tanh)
            dA_prev = np.matmul(W.T, dZ)

            # Apply dropout mask and scaling
            D_prev = cache["D" + str(layer - 1)]
            dA_prev = (dA_prev * D_prev) / keep_prob

            # Derivative of tanh
            A_prev_act = cache["A" + str(layer - 1)]
            dZ = dA_prev * (1 - A_prev_act ** 2)
