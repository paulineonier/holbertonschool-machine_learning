#!/usr/bin/env python3
"""Module that contains the deep_rnn function for forward propagation."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances of length l (layers).
        X (numpy.ndarray): Input data of shape (t, m, i).
            - t: Maximum number of time steps.
            - m: Batch size.
            - i: Dimensionality of the data.
        h_0 (numpy.ndarray): Initial hidden states of shape (l, m, h).
            - l: Number of layers.
            - h: Dimensionality of the hidden state.

    Returns:
        tuple: (H, Y)
            - H: numpy.ndarray of shape (t + 1, l, m, h) containing all hidden
              states across layers and time steps.
            - Y: numpy.ndarray of shape (t, m, o) containing all outputs of
              the last layer.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    # Initialize H with shape (t + 1, l, m, h)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    # Initialize Y with shape (t, m, o)
    Y = np.zeros((t, m, o))

    # Iterate over each time step
    for step in range(t):
        # Current input x_t starts as the data input for layer 0
        x_t = X[step]

        # Process layer by layer
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)

            # Store the computed hidden state for this layer
            H[step + 1, layer] = h_next

            # The current output becomes the input to the next layer
            x_t = h_next

        # Save output of the last layer at current time step
        Y[step] = y

    return H, Y
