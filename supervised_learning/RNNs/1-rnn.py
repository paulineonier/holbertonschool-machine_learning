#!/usr/bin/env python3
"""Module that contains the rnn function for forward propagation."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN over a sequence.

    Args:
        rnn_cell: Instance of RNNCell used for forward propagation.
        X (numpy.ndarray): Input data of shape (t, m, i).
            - t: Maximum number of time steps.
            - m: Batch size.
            - i: Dimensionality of the data.
        h_0 (numpy.ndarray): Initial hidden state of shape (m, h).
            - h: Dimensionality of the hidden state.

    Returns:
        tuple: (H, Y)
            - H: numpy.ndarray of shape (t + 1, m, h) containing all hidden
              states including the initial state h_0.
            - Y: numpy.ndarray of shape (t, m, o) containing all outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize H with room for h_0 + t hidden states
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Determine output dimensionality o by running a dummy output test
    # or referencing the output weights dimension of rnn_cell
    o = rnn_cell.Wy.shape[1]
    Y = np.zeros((t, m, o))

    # Iterate over all time steps
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
