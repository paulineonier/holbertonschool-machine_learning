#!/usr/bin/env python3
"""Module that computes policy gradients."""

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

    # Apply Softmax to convert values into probabilities
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient.

    Args:
        state: A matrix representing the current observation.
        weight: A matrix containing random weights.

    Returns:
        The selected action and the gradient.
    """
    # Reshape the state to a 2D matrix for the policy function
    state_matrix = state.reshape(1, -1)

    # Compute the probabilities for each possible action
    probabilities = policy(state_matrix, weight)[0]

    # Select an action randomly according to its probability
    action = np.random.choice(len(probabilities), p=probabilities)

    # Create a one-hot vector representing the selected action
    one_hot = np.zeros_like(probabilities)
    one_hot[action] = 1

    # Compute the gradient of the log-policy
    gradient = np.outer(state, one_hot - probabilities)

    # Return the selected action and the gradient
    return action, gradient
