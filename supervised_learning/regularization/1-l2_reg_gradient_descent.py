#!/usr/bin/env python3
"""
Module 1-l2_reg_gradient_descent
This module defines a function to update the weights and biases of a
neural network using gradient descent with L2 regularization.

Functions:
    l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
        Updates the weights and biases of a neural network in place
        using gradient descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Args:
        Y (numpy.ndarray): One-hot array of shape (classes, m) containing
                           the correct labels for the data.
        weights (dict): Dictionary of the weights and biases of the
                        neural network.
        cache (dict): Dictionary of the outputs of each layer of the
                      neural network.
        alpha (float): The learning rate.
        lambtha (float): The L2 regularization parameter.
        L (int): The number of layers of the neural network.

    Notes:
        - The neural network uses tanh activations on all layers except
          the last one, which uses a softmax activation.
        - The weights and biases are updated in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        # Compute gradients with L2 regularization
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update parameters
        weights['W' + str(l)] = W - alpha * dW
        weights['b' + str(l)] = b - alpha * db

        if l > 1:
            # Backprop through tanh activation
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.square(A_prev))
