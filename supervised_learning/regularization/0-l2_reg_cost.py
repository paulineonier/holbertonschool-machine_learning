#!/usr/bin/env python3
"""
Module 0-l2_reg_cost
This module defines a function to calculate the cost of a neural network
with L2 regularization applied.

Functions:
    l2_reg_cost(cost, lambtha, weights, L, m):
        Calculates the cost of a neural network with L2 regularization.
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (float): The cost of the network without L2 regularization.
        lambtha (float): The regularization parameter.
        weights (dict): A dictionary of the weights and biases of the
                        neural network. Keys are of the form 'Wl' or 'bl'
                        where l is the layer number.
        L (int): The number of layers in the neural network.
        m (int): The number of data points used.

    Returns:
        float: The cost of the network accounting for L2 regularization.
    """
    l2_norm = 0
    for i in range(1, L + 1):
        W_key = 'W' + str(i)
        l2_norm += np.sum(np.square(weights[W_key]))

    l2_cost = cost + (lambtha / (2 * m)) * l2_norm
    return l2_cost
