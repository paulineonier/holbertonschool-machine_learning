#!/usr/bin/env python3
"""Module that computes a policy using policy gradients."""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy using a weight matrix.

    Args:
        matrix: A matrix containing the current state.
        weight: A matrix containing the model weights.

    Returns:
        A matrix of probabilities for each possible action.
    """
    # Multiply the state matrix by the weight matrix
    z = np.matmul(matrix, weight)

    # Apply the exponential function to each value
    exp = np.exp(z)

    # Apply Softmax to convert the values into probabilities
    # axis=1 calculates the sum across each row
    # keepdims=True preserves the dimensions for broadcasting
    return exp / np.sum(exp, axis=1, keepdims=True)
